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
import requests
try:
    # Make aiohttp optional to avoid import errors in constrained environments/tests
    import aiohttp  # type: ignore
except Exception:
    aiohttp = None  # type: ignore
try:
    # Optional import: this module must be importable even if nfl_data_py is missing
    import nfl_data_py as nfl  # type: ignore
except Exception:
    nfl = None  # type: ignore
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Minimal required columns for snapshot CSVs (must stay in sync with docs/SNAPSHOT_SCHEMAS.md and CLI verification)
SNAPSHOT_MIN_COLUMNS: Dict[str, List[str]] = {
    # Foundation
    "schedules.csv": [
        "game_id","season","week","season_type","home_team","away_team",
        "kickoff_dt_utc","kickoff_dt_local","network","spread_close","total_close","officials_crew","stadium","roof_state"
    ],
    "rosters.csv": ["player_id","name","position","team","jersey_number","status","depth_chart_rank","snap_percentage","last_updated"],
    # Weekly
    "weekly_stats.csv": [
        "player_id","game_id","week","season","team","opponent","position",
        "passing_attempts","passing_completions","passing_yards","passing_touchdowns","interceptions",
        "rushing_attempts","rushing_yards","rushing_touchdowns",
        "targets","receptions","receiving_yards","receiving_touchdowns",
        "offensive_snaps","snap_percentage"
    ],
    "snaps.csv": [
        "player_id","game_id","team","position","offense_snaps","defense_snaps","st_snaps","offense_pct","defense_pct","st_pct"
    ],
    "pbp.csv": [
        "play_id","game_id","offense","defense","play_type","epa","success","air_yards","yac","pressure","blitz","personnel","formation"
    ],
    "weather.csv": [
        "game_id","stadium","temperature","humidity","wind_speed","wind_direction","precipitation","conditions","timestamp"
    ],
    # Roster/Status
    "depth_charts.csv": ["team","player_id","player_name","position","slot","role","package","depth_chart_rank","last_updated"],
    "injuries.csv": ["player_id","name","team","position","practice_status","game_status","designation","report_date","return_date"],
    # Odds
    "odds.csv": ["timestamp","book","market","player_id","team_id","line","over_odds","under_odds"],
    "odds_history.csv": ["ts_utc","book","market","selection_id","line","price","event_id","is_closing"],
    # Reference (for tests)
    "teams.csv": ["team_id","abbr","conference","division","coach","home_stadium_id"],
    "stadiums.csv": ["stadium_id","name","city","state","lat","lon","surface","roof","elevation"],
    "players.csv": [
        "player_id","name","birthdate","age","position","team",
        "height_inches","weight_lbs","dominant_hand","draft_year","draft_round","draft_pick"
    ],
}

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
    
    async def fetch_data(self, data_type: str = "weekly_stats", **kwargs) -> pd.DataFrame:
        """Fetch data from nfl_data_py - unified entry point"""
        season = kwargs.get('season', 2025)
        week = kwargs.get('week', None)
        
        if data_type == "rosters":
            return await self.fetch_rosters(season, week)
        elif data_type == "schedules":
            return await self.fetch_schedules(season)
        elif data_type == "weekly_stats":
            if week:
                return await self.fetch_weekly_stats(season, week)
            else:
                # Fetch current week or latest available
                return await self.fetch_weekly_stats(season, 1)
        elif data_type == "snap_counts":
            return await self.fetch_snap_counts(season, week or 1)
        else:
            logger.warning(f"Unknown data type: {data_type}")
            return pd.DataFrame()
    
    def _col(self, df: pd.DataFrame, name: str, default=None) -> pd.Series:
        """Return a column Series if present; otherwise a Series filled with default of proper length."""
        try:
            if name in df.columns:
                return df[name]
            # For nested keys that might be present under different names
            alt_map = {
                'player_name': ['player_display_name', 'name'],
                'team': ['recent_team', 'current_team'],
                'opponent_team': ['opponent', 'opp_team'],
                'gameday': ['game_date', 'gamedate'],
            }
            for alt in alt_map.get(name, []):
                if alt in df.columns:
                    return df[alt]
        except Exception:
            pass
        return pd.Series([default] * len(df))

    async def fetch_rosters(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch roster data"""
        date_str = f"{season}-W{week or 'current'}"
        
        if self.cache_manager.is_cached("rosters", date_str):
            cached_data = self.cache_manager.load_from_cache("rosters", date_str)
            if cached_data is not None:
                return cached_data
        
        try:
            if nfl is None:
                logger.warning("nfl_data_py not installed; returning empty rosters DataFrame")
                return pd.DataFrame()
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
            if nfl is None:
                logger.warning("nfl_data_py not installed; returning empty schedules DataFrame")
                return pd.DataFrame(columns=[
                    'game_id','season','week','season_type','home_team','away_team','kickoff_dt_utc',
                    'kickoff_dt_local','network','spread_close','total_close','officials_crew','stadium','roof_state'
                ])
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
            if nfl is None:
                logger.warning("nfl_data_py not installed; returning empty weekly stats DataFrame")
                return pd.DataFrame()
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
            if nfl is None:
                logger.warning("nfl_data_py not installed; returning empty snap counts DataFrame")
                return pd.DataFrame()
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
    
    async def fetch_depth_charts(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch depth chart data"""
        date_str = f"{season}-depth"
        try:
            if nfl is None:
                logger.warning("nfl_data_py not installed; returning empty depth charts DataFrame")
                return pd.DataFrame()
            logger.info(f"Fetching depth charts for {season}")
            # nfl_data_py depth charts API varies; attempt common endpoint
            depth = nfl.import_depth_charts([season]) if hasattr(nfl, 'import_depth_charts') else pd.DataFrame()
            if week and not depth.empty and 'week' in depth.columns:
                depth = depth[depth['week'] == week]
            normalized = self._normalize_depth_chart_data(depth)
            return normalized
        except Exception as e:
            logger.warning(f"Failed to fetch depth charts: {e}")
            return pd.DataFrame()
    
    async def fetch_injuries(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch injury reports"""
        try:
            if nfl is None:
                logger.warning("nfl_data_py not installed; returning empty injuries DataFrame")
                return pd.DataFrame()
            logger.info(f"Fetching injuries for {season}")
            injuries = nfl.import_injuries([season]) if hasattr(nfl, 'import_injuries') else pd.DataFrame()
            if week and not injuries.empty and 'week' in injuries.columns:
                injuries = injuries[injuries['week'] == week]
            normalized = self._normalize_injury_data(injuries)
            return normalized
        except Exception as e:
            logger.warning(f"Failed to fetch injuries: {e}")
            return pd.DataFrame()
    
    async def fetch_pbp(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch play-by-play data (lightweight subset)"""
        try:
            if nfl is None:
                logger.warning("nfl_data_py not installed; returning empty PBP DataFrame")
                return pd.DataFrame()
            logger.info(f"Fetching PBP for {season}")
            pbp = nfl.import_pbp_data([season]) if hasattr(nfl, 'import_pbp_data') else pd.DataFrame()
            if week and not pbp.empty and 'week' in pbp.columns:
                pbp = pbp[pbp['week'] == week]
            normalized = self._normalize_pbp_data(pbp)
            return normalized
        except Exception as e:
            logger.warning(f"Failed to fetch PBP: {e}")
            return pd.DataFrame()
    
    def _normalize_roster_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize roster data to our schema"""
        try:
            normalized = pd.DataFrame({
                'player_id': self._col(raw_data, 'player_id', ''),
                'name': self._col(raw_data, 'player_name', ''),
                'position': self._col(raw_data, 'position', ''),
                'team': self._col(raw_data, 'team', ''),
                'jersey_number': self._col(raw_data, 'jersey_number', None),
                'status': self._col(raw_data, 'status', 'active'),
                'depth_chart_rank': self._col(raw_data, 'depth_chart_order', None),
                'snap_percentage': pd.Series([None] * len(raw_data)),  # fill later
                'last_updated': pd.Series([datetime.now()] * len(raw_data))
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize roster data: {e}")
            return pd.DataFrame()
    
    def _normalize_schedule_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize schedule data to schema expected by tests (schedules.csv)."""
        try:
            gameday_series = self._col(raw_data, 'gameday', None)
            kickoff_dt = pd.to_datetime(gameday_series, errors='coerce')
            # Build DataFrame with exact headers expected by tests
            normalized = pd.DataFrame({
                'game_id': self._col(raw_data, 'game_id', ''),
                'season': self._col(raw_data, 'season', 0),
                'week': self._col(raw_data, 'week', 0),
                'season_type': self._col(raw_data, 'game_type', 'REG'),
                'home_team': self._col(raw_data, 'home_team', ''),
                'away_team': self._col(raw_data, 'away_team', ''),
                'kickoff_dt_utc': kickoff_dt.dt.strftime('%Y-%m-%dT%H:%M:%S').fillna(''),
                'kickoff_dt_local': kickoff_dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna(''),
                'network': pd.Series([''] * len(raw_data)),
                'spread_close': pd.Series([None] * len(raw_data)),
                'total_close': pd.Series([None] * len(raw_data)),
                'officials_crew': pd.Series([''] * len(raw_data)),
                'stadium': self._col(raw_data, 'stadium', ''),
                'roof_state': pd.Series([''] * len(raw_data)),
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize schedule data: {e}")
            # Return empty DataFrame with correct headers
            cols = [
                'game_id','season','week','season_type','home_team','away_team','kickoff_dt_utc',
                'kickoff_dt_local','network','spread_close','total_close','officials_crew','stadium','roof_state'
            ]
            return pd.DataFrame(columns=cols)
    
    def _normalize_stats_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize stats data to our schema"""
        try:
            normalized = pd.DataFrame({
                'player_id': self._col(raw_data, 'player_id', ''),
                'game_id': self._col(raw_data, 'game_id', ''),
                'week': self._col(raw_data, 'week', 0),
                'season': self._col(raw_data, 'season', 0),
                'team': self._col(raw_data, 'recent_team', ''),
                'opponent': self._col(raw_data, 'opponent_team', ''),
                'position': self._col(raw_data, 'position', ''),
                'passing_attempts': self._col(raw_data, 'passing_attempts', None),
                'passing_completions': self._col(raw_data, 'completions', None),
                'passing_yards': self._col(raw_data, 'passing_yards', None),
                'passing_touchdowns': self._col(raw_data, 'passing_tds', None),
                'interceptions': self._col(raw_data, 'interceptions', None),
                'rushing_attempts': self._col(raw_data, 'carries', None),
                'rushing_yards': self._col(raw_data, 'rushing_yards', None),
                'rushing_touchdowns': self._col(raw_data, 'rushing_tds', None),
                'targets': self._col(raw_data, 'targets', None),
                'receptions': self._col(raw_data, 'receptions', None),
                'receiving_yards': self._col(raw_data, 'receiving_yards', None),
                'receiving_touchdowns': self._col(raw_data, 'receiving_tds', None),
                'offensive_snaps': pd.Series([None] * len(raw_data)),
                'snap_percentage': pd.Series([None] * len(raw_data))
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize stats data: {e}")
            return pd.DataFrame()
    
    def _normalize_snap_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize snap count data to expected schema (snaps.csv)."""
        try:
            normalized = pd.DataFrame({
                'player_id': self._col(raw_data, 'player_id', ''),
                'game_id': self._col(raw_data, 'game_id', ''),
                'team': self._col(raw_data, 'team', ''),
                'position': self._col(raw_data, 'position', ''),
                'offense_snaps': self._col(raw_data, 'offense_snaps', None),
                'defense_snaps': self._col(raw_data, 'defense_snaps', None),
                'st_snaps': self._col(raw_data, 'st_snaps', None),
                'offense_pct': self._col(raw_data, 'offense_pct', None),
                'defense_pct': self._col(raw_data, 'defense_pct', None),
                'st_pct': self._col(raw_data, 'st_pct', None)
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize snap data: {e}")
            # Return empty DataFrame with the correct headers
            cols = [
                'player_id','game_id','team','position','offense_snaps','defense_snaps','st_snaps',
                'offense_pct','defense_pct','st_pct'
            ]
            return pd.DataFrame(columns=cols)
    
    def _normalize_depth_chart_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize depth chart data"""
        try:
            normalized = pd.DataFrame({
                'team': self._col(raw_data, 'team', ''),
                'player_id': self._col(raw_data, 'player_id', ''),
                'player_name': self._col(raw_data, 'player_name', ''),
                'position': self._col(raw_data, 'position', ''),
                'slot': self._col(raw_data, 'depth_position', ''),
                'role': self._col(raw_data, 'role', ''),
                'package': self._col(raw_data, 'package', ''),
                'depth_chart_rank': self._col(raw_data, 'depth_chart_order', None),
                'last_updated': pd.Series([datetime.now()] * len(raw_data))
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize depth chart data: {e}")
            return pd.DataFrame()
    
    def _normalize_injury_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize injuries data"""
        try:
            normalized = pd.DataFrame({
                'player_id': self._col(raw_data, 'player_id', ''),
                'name': self._col(raw_data, 'player_name', ''),
                'team': self._col(raw_data, 'team', ''),
                'position': self._col(raw_data, 'position', ''),
                'practice_status': self._col(raw_data, 'practice_status', ''),
                'game_status': self._col(raw_data, 'game_status', ''),
                'designation': self._col(raw_data, 'injury_status', ''),
                'report_date': pd.to_datetime(self._col(raw_data, 'report_date', None), errors='coerce'),
                'return_date': pd.to_datetime(self._col(raw_data, 'return_date', None), errors='coerce')
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize injury data: {e}")
            return pd.DataFrame()
    
    def _normalize_pbp_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize play-by-play to snapshot schema"""
        try:
            normalized = pd.DataFrame({
                'play_id': self._col(raw_data, 'play_id', ''),
                'game_id': self._col(raw_data, 'game_id', ''),
                'offense': self._col(raw_data, 'posteam', ''),
                'defense': self._col(raw_data, 'defteam', ''),
                'play_type': self._col(raw_data, 'play_type', ''),
                'epa': self._col(raw_data, 'epa', None),
                'success': self._col(raw_data, 'success', None),
                'air_yards': self._col(raw_data, 'air_yards', None),
                'yac': self._col(raw_data, 'yac_epa', None),
                'pressure': self._col(raw_data, 'qb_hit', ''),
                'blitz': self._col(raw_data, 'blitz', ''),
                'personnel': self._col(raw_data, 'personnel_off', ''),
                'formation': self._col(raw_data, 'offense_formation', '')
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize pbp data: {e}")
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
    
    async def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch weather data - unified entry point"""
        game_id = kwargs.get('game_id')
        stadium = kwargs.get('stadium', '')
        game_date = kwargs.get('game_date', datetime.now())
        
        if game_id and stadium:
            weather_data = await self.fetch_weather(game_id, stadium, game_date)
            if weather_data:
                return pd.DataFrame([asdict(weather_data)])
        
        return pd.DataFrame()
    
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
        # If aiohttp is not available, gracefully skip external weather fetch
        if aiohttp is None:
            logger.info("aiohttp not available; skipping NWS weather fetch")
            return None

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
    
    def _snapshot_dir(self, date_str: str) -> Path:
        """Return the snapshot directory path for a given date string."""
        path = Path("data/snapshots") / date_str
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _write_snapshot_csv(self, df: pd.DataFrame, filename: str, date_str: str) -> Path:
        """Write a DataFrame to the snapshot CSV file and return its path.
        If df is empty, still ensure the file exists with headers when possible.
        """
        snapshot_path = self._snapshot_dir(date_str) / filename
        try:
            min_cols = SNAPSHOT_MIN_COLUMNS.get(filename)
            if df is not None and hasattr(df, 'columns') and len(df.columns) > 0:
                # Write header, then overwrite with rows if present
                df.head(0).to_csv(snapshot_path, index=False)
                if not df.empty:
                    df.to_csv(snapshot_path, index=False)
            else:
                # No data/columns: write header-only if we know the minimal schema
                if min_cols:
                    pd.DataFrame(columns=min_cols).to_csv(snapshot_path, index=False)
                else:
                    # Fallback: create empty file
                    if not snapshot_path.exists():
                        snapshot_path.touch()
            logger.info(f"Snapshot written: {snapshot_path}")
        except Exception as e:
            logger.warning(f"Failed to write snapshot {snapshot_path}: {e}")
        return snapshot_path
    
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
            snapshot_date = datetime.now().strftime("%Y-%m-%d")
            # Fetch rosters
            rosters = await self.nfl_adapter.fetch_rosters(season, week)
            results['data_sources']['rosters'] = len(rosters)
            # Snapshot
            self._write_snapshot_csv(rosters, "rosters.csv", snapshot_date)
            
            # Fetch schedules
            schedules = await self.nfl_adapter.fetch_schedules(season)
            week_schedules = schedules[schedules['week'] == week] if len(schedules) > 0 else pd.DataFrame()
            results['data_sources']['schedules'] = len(week_schedules)
            self._write_snapshot_csv(week_schedules, "schedules.csv", snapshot_date)
            
            # Fetch weekly stats
            stats = await self.nfl_adapter.fetch_weekly_stats(season, week)
            results['data_sources']['stats'] = len(stats)
            # Optional snapshot (name 'weekly_stats.csv')
            self._write_snapshot_csv(stats, "weekly_stats.csv", snapshot_date)
            
            # Fetch snap counts
            snaps = await self.nfl_adapter.fetch_snap_counts(season, week)
            results['data_sources']['snap_counts'] = len(snaps)
            self._write_snapshot_csv(snaps, "snaps.csv", snapshot_date)
            
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
            # Snapshot weather
            try:
                if weather_data:
                    from dataclasses import asdict
                    weather_df = pd.DataFrame([asdict(w) for w in weather_data])
                else:
                    # Ensure header even if no weather data
                    weather_df = pd.DataFrame(columns=SNAPSHOT_MIN_COLUMNS["weather.csv"])  # type: ignore[index]
                self._write_snapshot_csv(weather_df, "weather.csv", snapshot_date)
            except Exception as e:
                logger.warning(f"Failed to snapshot weather data: {e}")
            
            # Depth charts
            depth = await self.nfl_adapter.fetch_depth_charts(season, week)
            results['data_sources']['depth_charts'] = len(depth)
            self._write_snapshot_csv(depth, "depth_charts.csv", snapshot_date)

            # Injuries
            injuries = await self.nfl_adapter.fetch_injuries(season, week)
            results['data_sources']['injuries'] = len(injuries)
            self._write_snapshot_csv(injuries, "injuries.csv", snapshot_date)

            # PBP
            pbp = await self.nfl_adapter.fetch_pbp(season, week)
            results['data_sources']['pbp'] = len(pbp)
            self._write_snapshot_csv(pbp, "pbp.csv", snapshot_date)

            # Additional snapshot placeholders to ensure schemas exist
            try:
                # Routes
                routes_cols = [
                    'season', 'week', 'team_id', 'player_id',
                    'routes_run', 'route_participation'
                ]
                self._write_snapshot_csv(pd.DataFrame(columns=routes_cols), "routes.csv", snapshot_date)

                # Usage shares
                usage_cols = [
                    'season', 'week', 'team_id', 'player_id',
                    'carry_share', 'target_share', 'rz_touch_share', 'gl_carry_share',
                    'pass_block_snaps', 'align_slot', 'align_wide', 'align_inline', 'align_backfield'
                ]
                self._write_snapshot_csv(pd.DataFrame(columns=usage_cols), "usage_shares.csv", snapshot_date)

                # Drives
                drives_cols = [
                    'drive_id', 'game_id', 'offense', 'start_q', 'start_clock', 'start_yardline',
                    'end_q', 'end_clock', 'result', 'plays', 'yards', 'time_elapsed', 'points'
                ]
                self._write_snapshot_csv(pd.DataFrame(columns=drives_cols), "drives.csv", snapshot_date)

                # Transactions
                txn_cols = ['date', 'team_id', 'player_id', 'type', 'detail']
                self._write_snapshot_csv(pd.DataFrame(columns=txn_cols), "transactions.csv", snapshot_date)

                # Inactives
                inactives_cols = ['season', 'week', 'team_id', 'player_id', 'pos', 'reason', 'declared_time']
                self._write_snapshot_csv(pd.DataFrame(columns=inactives_cols), "inactives.csv", snapshot_date)

                # Box score stats
                box_passing_cols = ['game_id', 'player_id', 'att', 'comp', 'yds', 'td', 'int', 'sacks', 'sack_yards', 'ypa', 'air_yards', 'aDOT', 'fumbles']
                box_rushing_cols = ['game_id', 'player_id', 'att', 'yds', 'td', 'long', 'ypc', 'fumbles']
                box_receiving_cols = ['game_id', 'player_id', 'targets', 'rec', 'yds', 'td', 'air_yards', 'yac', 'aDOT', 'drops', 'long']
                box_defense_cols = ['game_id', 'player_id', 'tackles', 'assists', 'sacks', 'tfl', 'qb_hits', 'ints', 'pbu', 'td']
                kicking_cols = ['game_id', 'player_id', 'fg_0_39', 'fg_40_49', 'fg_50p', 'fg_att', 'fg_made', 'xp_att', 'xp_made']
                self._write_snapshot_csv(pd.DataFrame(columns=box_passing_cols), "box_passing.csv", snapshot_date)
                self._write_snapshot_csv(pd.DataFrame(columns=box_rushing_cols), "box_rushing.csv", snapshot_date)
                self._write_snapshot_csv(pd.DataFrame(columns=box_receiving_cols), "box_receiving.csv", snapshot_date)
                self._write_snapshot_csv(pd.DataFrame(columns=box_defense_cols), "box_defense.csv", snapshot_date)
                self._write_snapshot_csv(pd.DataFrame(columns=kicking_cols), "kicking.csv", snapshot_date)

                # Team context and splits
                team_context_cols = ['season', 'week', 'team_id', 'opp_id', 'rest_days', 'travel_miles', 'tz_delta', 'pace_sn', 'pace_all', 'PROE', 'lead_pct', 'trail_pct', 'neutral_pct']
                team_splits_cols = ['season', 'week', 'team_id', 'pace_sn', 'pace_all', 'proe', 'rz_eff', 'g2g_eff', 'third_conv', 'fourth_att', 'vs_pos_rb_yds', 'vs_pos_wr_yds', 'vs_pos_te_yds']
                self._write_snapshot_csv(pd.DataFrame(columns=team_context_cols), "team_context.csv", snapshot_date)
                self._write_snapshot_csv(pd.DataFrame(columns=team_splits_cols), "team_splits.csv", snapshot_date)

                # Games metadata
                games_cols = ['game_id', 'roof_state', 'field_type', 'attendance', 'duration', 'closing_spread', 'closing_total']
                self._write_snapshot_csv(pd.DataFrame(columns=games_cols), "games.csv", snapshot_date)

                # Odds history
                odds_hist_cols = ['ts_utc', 'book', 'market', 'selection_id', 'line', 'price', 'event_id', 'is_closing']
                self._write_snapshot_csv(pd.DataFrame(columns=odds_hist_cols), "odds_history.csv", snapshot_date)
                # Current odds snapshot placeholder
                self._write_snapshot_csv(pd.DataFrame(columns=SNAPSHOT_MIN_COLUMNS["odds.csv"]), "odds.csv", snapshot_date)  # type: ignore[index]
            except Exception as e:
                logger.warning(f"Failed to write one or more placeholder CSVs: {e}")

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
