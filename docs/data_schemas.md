# NFL Data Schemas Documentation

This document defines the standardized data schemas used throughout the NFL Betting Analyzer system.

## Data Flow Architecture

```
Raw Sources → Ingestion Adapters → Cached Snapshots → Validation → Database
     ↓              ↓                    ↓              ↓          ↓
NFL Data Py    Schema Mapping      data/snapshots/  Quality    SQLAlchemy
Weather API    Type Conversion     YYYY-MM-DD/      Checks     Models
Odds API       Error Handling      *.parquet        Logging    
```

## Core Data Schemas

### PlayerRoster
Standardized player roster information across all sources.

```python
@dataclass
class PlayerRoster:
    player_id: str              # Unique NFL player identifier
    name: str                   # Full player name
    position: str               # Position code (QB, RB, WR, TE, etc.)
    team: str                   # Team abbreviation (3 chars)
    jersey_number: Optional[int] # Jersey number
    status: str                 # active, inactive, injured_reserve, practice_squad
    depth_chart_rank: Optional[int] # 1=starter, 2=backup, etc.
    snap_percentage: Optional[float] # % of team snaps (0.0-1.0)
    last_updated: datetime      # When this record was last updated
```

**Source Mapping:**
- `nfl_data_py.import_rosters()` → PlayerRoster
- Cached as: `data/snapshots/YYYY-MM-DD/rosters.parquet`

### GameSchedule
Game scheduling and venue information.

```python
@dataclass
class GameSchedule:
    game_id: str                # Unique game identifier
    season: int                 # Season year
    week: int                   # Week number (1-18 regular, 19+ playoffs)
    game_date: datetime         # Game start time (UTC)
    home_team: str              # Home team abbreviation
    away_team: str              # Away team abbreviation
    stadium: str                # Stadium name
    weather_conditions: Optional[Dict[str, Any]] # Weather data reference
    game_status: str            # scheduled, in_progress, completed, postponed
```

**Source Mapping:**
- `nfl_data_py.import_schedules()` → GameSchedule
- Cached as: `data/snapshots/YYYY-schedule/schedules.parquet`

### PlayerStats
Comprehensive player statistics for a single game.

```python
@dataclass
class PlayerStats:
    player_id: str              # Links to PlayerRoster.player_id
    game_id: str                # Links to GameSchedule.game_id
    week: int                   # Week number
    season: int                 # Season year
    team: str                   # Player's team
    opponent: str               # Opponent team
    position: str               # Position played
    
    # Passing Statistics
    passing_attempts: Optional[int]
    passing_completions: Optional[int]
    passing_yards: Optional[int]
    passing_touchdowns: Optional[int]
    interceptions: Optional[int]
    
    # Rushing Statistics
    rushing_attempts: Optional[int]
    rushing_yards: Optional[int]
    rushing_touchdowns: Optional[int]
    
    # Receiving Statistics
    targets: Optional[int]
    receptions: Optional[int]
    receiving_yards: Optional[int]
    receiving_touchdowns: Optional[int]
    
    # Snap Counts
    offensive_snaps: Optional[int]
    snap_percentage: Optional[float]
```

**Source Mapping:**
- `nfl_data_py.import_weekly_data()` → PlayerStats (base stats)
- `nfl_data_py.import_snap_counts()` → PlayerStats (snap data merged)
- Cached as: `data/snapshots/YYYY-MM-DD/weekly_stats.parquet`

### WeatherData
Weather conditions for outdoor games.

```python
@dataclass
class WeatherData:
    game_id: str                # Links to GameSchedule.game_id
    stadium: str                # Stadium name
    temperature: Optional[float] # Temperature in Fahrenheit
    humidity: Optional[float]    # Humidity percentage (0-100)
    wind_speed: Optional[float]  # Wind speed in MPH
    wind_direction: Optional[str] # Wind direction (N, NE, E, etc.)
    precipitation: Optional[float] # Precipitation in inches
    conditions: str             # clear, cloudy, rain, snow, dome
    timestamp: datetime         # When weather was recorded
```

**Source Mapping:**
- National Weather Service API → WeatherData
- Domed stadiums get synthetic "dome" conditions
- Cached as: `data/snapshots/YYYY-MM-DD/weather_{game_id}.parquet`

### InjuryReport
Player injury status and reports.

```python
@dataclass
class InjuryReport:
    player_id: str              # Links to PlayerRoster.player_id
    name: str                   # Player name
    team: str                   # Team abbreviation
    position: str               # Position
    injury_status: str          # out, doubtful, questionable, probable
    injury_description: str     # Body part and injury type
    report_date: datetime       # When report was issued
    week: int                   # Week number
    season: int                 # Season year
```

**Source Mapping:**
- `nfl_data_py.import_injuries()` → InjuryReport
- Cached as: `data/snapshots/YYYY-MM-DD/injuries.parquet`

### BettingLine
Sportsbook betting lines and odds.

```python
@dataclass
class BettingLine:
    game_id: str                # Links to GameSchedule.game_id
    market_type: str            # spread, total, moneyline, player_props
    player_id: Optional[str]    # For player prop bets
    line_value: float           # Point spread, total points, or prop value
    odds: int                   # American odds format (-110, +150, etc.)
    sportsbook: str             # Sportsbook identifier
    timestamp: datetime         # When line was recorded
```

**Source Mapping:**
- The Odds API → BettingLine
- Multiple sportsbooks aggregated
- Cached as: `data/snapshots/YYYY-MM-DD/betting_lines.parquet`

## Caching Strategy

### Directory Structure
```
data/snapshots/
├── 2024-09-08/                 # Daily snapshots
│   ├── rosters.parquet
│   ├── weekly_stats.parquet
│   ├── snap_counts.parquet
│   ├── injuries.parquet
│   ├── weather_2024090800.parquet
│   └── betting_lines.parquet
├── 2024-schedule/              # Season-long data
│   └── schedules.parquet
└── schemas/                    # Schema documentation
    └── schema_versions.json
```

### Cache Invalidation Rules
- **Rosters**: 6 hours (roster changes during week)
- **Schedules**: 24 hours (rarely change mid-season)
- **Weekly Stats**: 1 hour during games, 6 hours otherwise
- **Weather**: 12 hours (forecasts update regularly)
- **Betting Lines**: 15 minutes (lines move frequently)
- **Injuries**: 2 hours (reports update throughout day)

### Data Quality Validation

Each schema includes validation rules:

1. **Required Fields**: All non-Optional fields must be present
2. **Type Validation**: Ensure correct data types
3. **Range Validation**: Numeric fields within expected ranges
4. **Referential Integrity**: Foreign keys must exist
5. **Business Logic**: Position-specific stat validation

### Error Handling

```python
class DataValidationError(Exception):
    """Raised when data fails schema validation"""
    pass

class CacheError(Exception):
    """Raised when cache operations fail"""
    pass

class IngestionError(Exception):
    """Raised when data ingestion fails"""
    pass
```

## Integration with Existing Systems

### Database Models
The schemas map to SQLAlchemy models in `core/database_models.py`:
- PlayerRoster → Player table
- GameSchedule → Game table  
- PlayerStats → PlayerGameStats table
- WeatherData → GameWeather table (new)
- BettingLine → BettingLines table (new)

### Feature Engineering
Cached data feeds into `core/models/feature_engineering.py`:
- Snap percentages → Usage features
- Weather conditions → Environmental features
- Injury reports → Availability features
- Historical stats → Trend features

### Model Training
Clean, validated data ensures model quality:
- Consistent player identification
- Accurate snap count filtering
- Weather-adjusted performance metrics
- Injury-aware projections

## Usage Examples

### Fetch and Cache Weekly Data
```python
from core.data.ingestion_adapters import UnifiedDataIngestion

ingestion = UnifiedDataIngestion(session)
results = await ingestion.ingest_weekly_data(2024, 1)
```

### Load Cached Data
```python
from core.data.ingestion_adapters import CacheManager

cache = CacheManager()
rosters = cache.load_from_cache("rosters", "2024-09-08")
```

### Validate Schema
```python
from core.data.ingestion_adapters import PlayerRoster, NFLDataPyAdapter

adapter = NFLDataPyAdapter(cache_manager)
data = await adapter.fetch_rosters(2024, 1)
is_valid = adapter.validate_schema(data, PlayerRoster)
```

This schema system ensures data consistency, enables efficient caching, and provides clear documentation for all data transformations in the pipeline.
