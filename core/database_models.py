"""
NFL Betting Analyzer - Unified Database Models
Authoritative database schema combining enhanced features with existing data compatibility.
Version 2.0 - 2025 Season Ready
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    JSON, DECIMAL, Date, Time, create_engine
)
from sqlalchemy.orm import relationship, Mapped, mapped_column, sessionmaker, DeclarativeBase
from sqlalchemy.sql import func
from datetime import datetime, date, time
from typing import Optional, Dict, Any, List
import json

class Base(DeclarativeBase):
    """Typed declarative base for SQLAlchemy 2.0 style models (mypy-friendly)."""
    pass


class Player(Base):
    """Core player information - unified schema for 2025 season."""
    __tablename__ = 'players'
    
    # Primary Key
    player_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    
    # Basic Info (required fields)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(50))
    last_name: Mapped[Optional[str]] = mapped_column(String(50))
    position: Mapped[str] = mapped_column(String(10), nullable=False)
    current_team: Mapped[Optional[str]] = mapped_column(String(10))
    
    # Status tracking (critical for 2025 accuracy)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_retired: Mapped[bool] = mapped_column(Boolean, default=False)
    retirement_date: Mapped[Optional[date]] = mapped_column(Date)
    
    # Physical Attributes
    height_inches: Mapped[Optional[int]] = mapped_column(Integer)
    weight_lbs: Mapped[Optional[int]] = mapped_column(Integer)
    age: Mapped[Optional[int]] = mapped_column(Integer)
    birth_date: Mapped[Optional[date]] = mapped_column(Date)
    
    # Draft Info
    draft_year: Mapped[Optional[int]] = mapped_column(Integer)
    draft_round: Mapped[Optional[int]] = mapped_column(Integer)
    draft_pick: Mapped[Optional[int]] = mapped_column(Integer)
    college: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Experience
    years_experience: Mapped[Optional[int]] = mapped_column(Integer)
    seasons_played: Mapped[Optional[int]] = mapped_column(Integer)
    
    # External IDs for data integration
    espn_id: Mapped[Optional[str]] = mapped_column(String(50))
    yahoo_id: Mapped[Optional[str]] = mapped_column(String(50))
    sleeper_id: Mapped[Optional[str]] = mapped_column(String(50))
    pfr_id: Mapped[Optional[str]] = mapped_column(String(50))
    gsis_id: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Role and depth chart
    role_classification: Mapped[Optional[str]] = mapped_column(String(20))
    depth_chart_rank: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Data quality tracking
    data_quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    last_validated: Mapped[Optional[datetime]] = mapped_column(DateTime)
    inconsistency_flags: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    game_stats = relationship("PlayerGameStats", back_populates="player", cascade="all, delete-orphan")
    predictions = relationship("PlayerPrediction", back_populates="player", cascade="all, delete-orphan")


class BettingLine(Base):
    """Minimal betting line model for imports and basic storage."""
    __tablename__ = 'betting_lines'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey('games.game_id'))
    player_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey('players.player_id'))
    team: Mapped[Optional[str]] = mapped_column(String(10))
    market: Mapped[Optional[str]] = mapped_column(String(50))
    book: Mapped[Optional[str]] = mapped_column(String(50))
    line: Mapped[Optional[float]] = mapped_column(Float)
    over_odds: Mapped[Optional[int]] = mapped_column(Integer)
    under_odds: Mapped[Optional[int]] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class FeatureStore(Base):
    """Minimal feature store to persist engineered features per player-game."""
    __tablename__ = 'feature_store'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(String(50), ForeignKey('players.player_id'), index=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey('games.game_id'), index=True)
    position: Mapped[Optional[str]] = mapped_column(String(10))

    # Store feature categories as JSON strings for portability
    recent_form_features: Mapped[Optional[str]] = mapped_column(Text)
    seasonal_features: Mapped[Optional[str]] = mapped_column(Text)
    opponent_features: Mapped[Optional[str]] = mapped_column(Text)
    contextual_features: Mapped[Optional[str]] = mapped_column(Text)
    advanced_features: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('player_id', 'game_id', name='uq_feature_store_player_game'),
    )


class HistoricalValidationReport(Base):
    """Detailed validation report records for standardized historical data."""
    __tablename__ = 'historical_validation_reports'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[Optional[int]] = mapped_column(Integer)
    report_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


# --- Phase 2: Minimal tables for routes and usage shares ---

class PlayerRoutes(Base):
    """Weekly player route participation."""
    __tablename__ = 'player_routes'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    team_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    player_id: Mapped[str] = mapped_column(String(50), ForeignKey('players.player_id'), nullable=False, index=True)
    routes_run: Mapped[Optional[int]] = mapped_column(Integer)
    route_participation: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint('season', 'week', 'team_id', 'player_id', name='uq_player_routes_key'),
    )


class UsageShares(Base):
    """Weekly usage share metrics per player."""
    __tablename__ = 'usage_shares'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    team_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    player_id: Mapped[str] = mapped_column(String(50), ForeignKey('players.player_id'), nullable=False, index=True)
    carry_share: Mapped[Optional[float]] = mapped_column(Float)
    target_share: Mapped[Optional[float]] = mapped_column(Float)
    rz_touch_share: Mapped[Optional[float]] = mapped_column(Float)
    gl_carry_share: Mapped[Optional[float]] = mapped_column(Float)
    pass_block_snaps: Mapped[Optional[int]] = mapped_column(Integer)
    align_slot: Mapped[Optional[float]] = mapped_column(Float)
    align_wide: Mapped[Optional[float]] = mapped_column(Float)
    align_inline: Mapped[Optional[float]] = mapped_column(Float)
    align_backfield: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint('season', 'week', 'team_id', 'player_id', name='uq_usage_shares_key'),
    )


class Team(Base):
    """NFL Team information for 2025 season."""
    __tablename__ = 'teams'
    
    team_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    team_name: Mapped[str] = mapped_column(String(50), nullable=False)
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    conference: Mapped[str] = mapped_column(String(10), nullable=False)
    division: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Stadium Info
    stadium_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)
    surface_type: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Colors and branding
    primary_color: Mapped[Optional[str]] = mapped_column(String(20))
    secondary_color: Mapped[Optional[str]] = mapped_column(String(20))
    logo_url: Mapped[Optional[str]] = mapped_column(String(255))
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class Game(Base):
    """NFL Games and scheduling for 2025 season."""
    __tablename__ = 'games'
    
    game_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    
    # Game Details
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    game_type: Mapped[str] = mapped_column(String(20), nullable=False, default='REG')
    game_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Teams
    home_team: Mapped[str] = mapped_column(String(10), nullable=False)
    away_team: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Scores (nullable before completion)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Game Status
    game_status: Mapped[str] = mapped_column(String(20), default='scheduled')
    is_overtime: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Weather and conditions
    stadium: Mapped[Optional[str]] = mapped_column(String(100))
    roof_state: Mapped[Optional[str]] = mapped_column(String(20))
    surface: Mapped[Optional[str]] = mapped_column(String(20))
    weather_temperature: Mapped[Optional[int]] = mapped_column(Integer)
    weather_humidity: Mapped[Optional[int]] = mapped_column(Integer)
    weather_wind_speed: Mapped[Optional[int]] = mapped_column(Integer)
    weather_conditions: Mapped[Optional[str]] = mapped_column(String(50))
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    player_stats = relationship("PlayerGameStats", back_populates="game")
    predictions = relationship("GamePrediction", back_populates="game")
    
    __table_args__ = (
        Index('idx_game_season_week', 'season', 'week'),
        Index('idx_game_teams', 'home_team', 'away_team'),
        Index('idx_game_date', 'game_date'),
        UniqueConstraint('season', 'week', 'home_team', 'away_team'),
    )


class ModelPerformance(Base):
    """Store performance metrics for trained models (stub for import compatibility)."""
    __tablename__ = 'model_performance'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Common fields across usages
    model_key: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    model_type: Mapped[Optional[str]] = mapped_column(String(50))
    evaluation_period: Mapped[Optional[str]] = mapped_column(String(100))
    # Metrics
    r2_score: Mapped[Optional[float]] = mapped_column(Float)
    rmse: Mapped[Optional[float]] = mapped_column(Float)
    mae: Mapped[Optional[float]] = mapped_column(Float)
    cv_score: Mapped[Optional[float]] = mapped_column(Float)
    accuracy_metrics: Mapped[Optional[str]] = mapped_column(Text)  # JSON serialized
    sample_count: Mapped[Optional[int]] = mapped_column(Integer)
    total_predictions: Mapped[Optional[int]] = mapped_column(Integer)
    evaluation_start_date: Mapped[Optional[date]] = mapped_column(Date)
    evaluation_end_date: Mapped[Optional[date]] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class PlayerGameStats(Base):
    """Player statistics for each game - compatible with existing data."""
    __tablename__ = 'player_game_stats'
    
    # Primary Key
    stat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    player_id: Mapped[str] = mapped_column(String(50), ForeignKey('players.player_id'), nullable=False)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey('games.game_id'), nullable=False)
    
    # Game Context
    team: Mapped[str] = mapped_column(String(10), nullable=False)
    opponent: Mapped[str] = mapped_column(String(10), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    
    # Passing Stats
    passing_attempts: Mapped[int] = mapped_column(Integer, default=0)
    passing_completions: Mapped[int] = mapped_column(Integer, default=0)
    passing_yards: Mapped[int] = mapped_column(Integer, default=0)
    passing_touchdowns: Mapped[int] = mapped_column(Integer, default=0)
    passing_interceptions: Mapped[int] = mapped_column(Integer, default=0)
    passing_sacks: Mapped[int] = mapped_column(Integer, default=0)
    passing_sack_yards: Mapped[int] = mapped_column(Integer, default=0)
    
    # Rushing Stats
    rushing_attempts: Mapped[int] = mapped_column(Integer, default=0)
    rushing_yards: Mapped[int] = mapped_column(Integer, default=0)
    rushing_touchdowns: Mapped[int] = mapped_column(Integer, default=0)
    rushing_fumbles: Mapped[int] = mapped_column(Integer, default=0)
    rushing_first_downs: Mapped[int] = mapped_column(Integer, default=0)
    
    # Receiving Stats
    targets: Mapped[int] = mapped_column(Integer, default=0)
    receptions: Mapped[int] = mapped_column(Integer, default=0)
    receiving_yards: Mapped[int] = mapped_column(Integer, default=0)
    receiving_touchdowns: Mapped[int] = mapped_column(Integer, default=0)
    receiving_fumbles: Mapped[int] = mapped_column(Integer, default=0)
    receiving_first_downs: Mapped[int] = mapped_column(Integer, default=0)
    
    # Advanced Stats
    snap_count: Mapped[Optional[int]] = mapped_column(Integer)
    snap_percentage: Mapped[Optional[float]] = mapped_column(Float)
    routes_run: Mapped[Optional[int]] = mapped_column(Integer)
    air_yards: Mapped[Optional[int]] = mapped_column(Integer)
    yards_after_catch: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Fantasy Points (multiple scoring systems)
    fantasy_points_standard: Mapped[float] = mapped_column(Float, default=0.0)
    fantasy_points_ppr: Mapped[float] = mapped_column(Float, default=0.0)
    fantasy_points_half_ppr: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Data validation
    stats_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    role_at_time: Mapped[Optional[str]] = mapped_column(String(20))
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="game_stats")
    game = relationship("Game", back_populates="player_stats")
    
    __table_args__ = (
        Index('idx_player_game_stats_player', 'player_id'),
        Index('idx_player_game_stats_game', 'game_id'),
        Index('idx_player_game_stats_team', 'team'),
        UniqueConstraint('player_id', 'game_id'),
    )


class PlayerPrediction(Base):
    """ML model predictions for player performance."""
    __tablename__ = 'player_predictions'
    
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # References
    player_id: Mapped[str] = mapped_column(String(50), ForeignKey('players.player_id'), nullable=False)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey('games.game_id'), nullable=False)
    
    # Model Info
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Predictions
    predicted_passing_yards: Mapped[Optional[float]] = mapped_column(Float)
    predicted_passing_tds: Mapped[Optional[float]] = mapped_column(Float)
    predicted_rushing_yards: Mapped[Optional[float]] = mapped_column(Float)
    predicted_rushing_tds: Mapped[Optional[float]] = mapped_column(Float)
    predicted_receiving_yards: Mapped[Optional[float]] = mapped_column(Float)
    predicted_receiving_tds: Mapped[Optional[float]] = mapped_column(Float)
    predicted_receptions: Mapped[Optional[float]] = mapped_column(Float)
    predicted_fantasy_points: Mapped[Optional[float]] = mapped_column(Float)
    
    # Confidence intervals
    confidence_overall: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_passing: Mapped[Optional[float]] = mapped_column(Float)
    confidence_rushing: Mapped[Optional[float]] = mapped_column(Float)
    confidence_receiving: Mapped[Optional[float]] = mapped_column(Float)
    
    # Prediction metadata
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    player = relationship("Player", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_player_game', 'player_id', 'game_id'),
        Index('idx_prediction_timestamp', 'prediction_timestamp'),
        UniqueConstraint('player_id', 'game_id', 'model_version'),
    )


class GamePrediction(Base):
    """Game outcome predictions."""
    __tablename__ = 'game_predictions'
    
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey('games.game_id'), nullable=False)
    
    # Model Info
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Predictions
    predicted_home_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_away_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_total_points: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_spread: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Win probabilities
    home_win_probability: Mapped[float] = mapped_column(Float, nullable=False)
    away_win_probability: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Confidence
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    game = relationship("Game", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_game_prediction_game', 'game_id'),
        Index('idx_game_prediction_timestamp', 'prediction_timestamp'),
        UniqueConstraint('game_id', 'model_version'),
    )


# --- Additional models required by historical standardizer/tests ---

class DepthChart(Base):
    """Team depth chart entries with ranks for each position."""
    __tablename__ = 'depth_chart'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    position: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    player_id: Mapped[str] = mapped_column(String(50), ForeignKey('players.player_id'), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[Optional[int]] = mapped_column(Integer)
    season: Mapped[Optional[int]] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_depth_team_pos', 'team', 'position'),
        Index('idx_depth_team_week', 'team', 'season', 'week'),
    )

class PlayerIdentityMapping(Base):
    """Mapping from original player IDs to master standardized IDs."""
    __tablename__ = 'player_identity_mappings'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    original_player_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    master_player_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0)
    resolution_method: Mapped[str] = mapped_column(String(50), default='automated')
    canonical_name: Mapped[Optional[str]] = mapped_column(String(100))
    primary_position: Mapped[Optional[str]] = mapped_column(String(10))
    seasons_active: Mapped[Optional[str]] = mapped_column(Text)
    teams_played_for: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_identity_original', 'original_player_id'),
        Index('idx_identity_master', 'master_player_id'),
    )


class StatTerminologyMapping(Base):
    """Mapping between original stat column names and standardized names by season."""
    __tablename__ = 'stat_terminology_mappings'

    mapping_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    standard_stat_name: Mapped[str] = mapped_column(String(100), nullable=False)
    original_column_name: Mapped[str] = mapped_column(String(100), nullable=False)
    mapping_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    mapping_method: Mapped[str] = mapped_column(String(50), default='automated')
    stat_category: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_stat_map_season', 'season'),
        Index('idx_stat_map_standard', 'standard_stat_name'),
    )


class HistoricalDataStandardization(Base):
    """Summary records for historical data standardization runs per season."""
    __tablename__ = 'historical_data_standardization'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    records_standardized: Mapped[int] = mapped_column(Integer, default=0)
    data_quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    player_mappings_created: Mapped[int] = mapped_column(Integer, default=0)
    terminology_mappings_created: Mapped[int] = mapped_column(Integer, default=0)
    validation_report: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[Optional[str]] = mapped_column(String(50))
    issues_found: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


def create_all_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(engine)


def get_db_session(database_url: str = "sqlite:///nfl_predictions.db"):
    """Get database session. Ensures tables are created for empty/new DBs."""
    engine = create_engine(database_url)
    try:
        Base.metadata.create_all(engine)
        # Apply lightweight migrations for new columns when running against an existing DB
        try:
            migrate_database(engine)
        except Exception:
            # Migrations are best-effort during session creation
            pass
    except Exception:
        pass
    Session = sessionmaker(bind=engine)
    return Session()


def migrate_database(engine):
    """Migrate existing database to new schema."""
    from sqlalchemy import text
    import logging
    logger = logging.getLogger(__name__)
    
    def column_exists(conn, table_name, column_name):
        """Check if column exists in table."""
        try:
            conn.execute(text(f"SELECT {column_name} FROM {table_name} LIMIT 1"))
            return True
        except:
            return False
    
    with engine.connect() as conn:
        # Add missing columns to players table
        player_columns = [
            ("is_retired", "BOOLEAN DEFAULT 0"),
            ("retirement_date", "DATE"),
            ("years_experience", "INTEGER"),
            ("draft_year", "INTEGER"),
            ("draft_round", "INTEGER"),
            ("draft_pick", "INTEGER"),
            ("college", "VARCHAR(100)"),
            ("height_inches", "INTEGER"),
            ("weight_lbs", "INTEGER"),
            ("birthdate", "DATE"),
            ("nfl_id", "VARCHAR(20)"),
            ("espn_id", "VARCHAR(20)"),
            ("yahoo_id", "VARCHAR(20)"),
            ("sleeper_id", "VARCHAR(20)"),
            ("rotowire_id", "VARCHAR(20)"),
            ("role_primary", "VARCHAR(20)"),
            ("role_secondary", "VARCHAR(20)"),
            ("injury_status", "VARCHAR(20)"),
            ("injury_description", "TEXT"),
            ("data_quality_score", "FLOAT DEFAULT 100.0"),
            ("last_validated", "TIMESTAMP"),
            ("notes", "TEXT")
        ]
        
        for column_name, column_def in player_columns:
            if not column_exists(conn, "players", column_name):
                try:
                    conn.execute(text(f"ALTER TABLE players ADD COLUMN {column_name} {column_def}"))
                    conn.commit()
                    logger.info(f"Added column {column_name} to players table")
                except Exception as e:
                    logger.debug(f"Could not add column {column_name}: {e}")
                    pass
        
        # Add missing columns to player_game_stats table
        stats_columns = [
            ("stats_validated", "BOOLEAN DEFAULT 0"),
            ("game_date", "DATE"),
            ("season", "INTEGER"),
            ("role_at_time", "VARCHAR(20)"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP")
        ]
        
        for column_name, column_def in stats_columns:
            if not column_exists(conn, "player_game_stats", column_name):
                try:
                    conn.execute(text(f"ALTER TABLE player_game_stats ADD COLUMN {column_name} {column_def}"))
                    conn.commit()
                    logger.info(f"Added column {column_name} to player_game_stats table")
                except Exception as e:
                    logger.debug(f"Could not add column {column_name}: {e}")
                    pass

        # Ensure new weather/venue columns exist on games table
        games_columns = [
            ("roof_state", "VARCHAR(20)"),
            ("surface", "VARCHAR(20)"),
        ]
        for column_name, column_def in games_columns:
            if not column_exists(conn, "games", column_name):
                try:
                    conn.execute(text(f"ALTER TABLE games ADD COLUMN {column_name} {column_def}"))
                    conn.commit()
                    logger.info(f"Added column {column_name} to games table")
                except Exception as e:
                    logger.debug(f"Could not add column {column_name} on games: {e}")
                    pass


def validate_player_status():
    """Validate that no retired players are marked as active."""
    session = get_db_session()
    
    try:
        # Find players who should be retired but are marked active
        retired_players = ['Tom Brady', 'Matt Ryan', 'Ben Roethlisberger', 'Rob Gronkowski']
        
        for name in retired_players:
            player = session.query(Player).filter(Player.name.ilike(f'%{name}%')).first()
            if player and player.is_active:
                player.is_active = False
                if hasattr(player, 'is_retired'):
                    player.is_retired = True
                    player.retirement_date = date(2024, 12, 31)
                
        session.commit()
    finally:
        session.close()


# Backward compatibility aliases
# Model __tablename__ values already match expected table names, so no renaming is required.
