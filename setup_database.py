"""
Modern NFL Database Models using SQLAlchemy 2.0
Optimized for player performance and game prediction tasks.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    JSON, DECIMAL, Date, Time
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime, date, time
from typing import Optional, Dict, Any, List
import json

Base = declarative_base()


class Player(Base):
    """Core player information and metadata."""
    __tablename__ = 'players'
    
    # Primary Key
    player_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    
    # Basic Info
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(50))
    last_name: Mapped[Optional[str]] = mapped_column(String(50))
    position: Mapped[str] = mapped_column(String(10), nullable=False)
    current_team: Mapped[Optional[str]] = mapped_column(String(10))
    
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
    
    # External IDs
    espn_id: Mapped[Optional[str]] = mapped_column(String(50))
    yahoo_id: Mapped[Optional[str]] = mapped_column(String(50))
    sleeper_id: Mapped[Optional[str]] = mapped_column(String(50))
    pfr_id: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    game_stats = relationship("PlayerGameStats", back_populates="player")
    predictions = relationship("PlayerPrediction", back_populates="player")
    
    __table_args__ = (
        Index('idx_player_position', 'position'),
        Index('idx_player_team', 'current_team'),
        Index('idx_player_active', 'is_active'),
    )


class Team(Base):
    """NFL Team information."""
    __tablename__ = 'teams'
    
    team_id: Mapped[str] = mapped_column(String(10), primary_key=True)  # e.g., 'KC', 'BUF'
    team_name: Mapped[str] = mapped_column(String(50), nullable=False)
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    conference: Mapped[str] = mapped_column(String(10), nullable=False)  # AFC/NFC
    division: Mapped[str] = mapped_column(String(10), nullable=False)    # North/South/East/West
    
    # Stadium Info
    stadium_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)
    surface_type: Mapped[Optional[str]] = mapped_column(String(20))  # Grass/Turf
    
    # Colors and branding
    primary_color: Mapped[Optional[str]] = mapped_column(String(20))
    secondary_color: Mapped[Optional[str]] = mapped_column(String(20))
    logo_url: Mapped[Optional[str]] = mapped_column(String(255))
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class Game(Base):
    """NFL Games and scheduling."""
    __tablename__ = 'games'
    
    game_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    
    # Game Details
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    game_type: Mapped[str] = mapped_column(String(20), nullable=False)  # REG/WC/DIV/CONF/SB
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    game_time: Mapped[Optional[time]] = mapped_column(Time)
    
    # Teams
    home_team: Mapped[str] = mapped_column(String(10), ForeignKey('teams.team_id'), nullable=False)
    away_team: Mapped[str] = mapped_column(String(10), ForeignKey('teams.team_id'), nullable=False)
    
    # Scores (nullable before game completion)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Game Status
    game_status: Mapped[str] = mapped_column(String(20), default='scheduled')  # scheduled/in_progress/completed/postponed
    is_overtime: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Stadium and Conditions
    stadium: Mapped[Optional[str]] = mapped_column(String(100))
    weather_temperature: Mapped[Optional[int]] = mapped_column(Integer)  # Fahrenheit
    weather_humidity: Mapped[Optional[int]] = mapped_column(Integer)     # Percentage
    weather_wind_speed: Mapped[Optional[int]] = mapped_column(Integer)   # MPH
    weather_conditions: Mapped[Optional[str]] = mapped_column(String(50)) # Clear/Rain/Snow/etc
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    player_stats = relationship("PlayerGameStats", back_populates="game")
    betting_lines = relationship("BettingLine", back_populates="game")
    predictions = relationship("GamePrediction", back_populates="game")
    
    __table_args__ = (
        Index('idx_game_season_week', 'season', 'week'),
        Index('idx_game_teams', 'home_team', 'away_team'),
        Index('idx_game_date', 'game_date'),
        UniqueConstraint('season', 'week', 'home_team', 'away_team'),
    )


class PlayerGameStats(Base):
    """Comprehensive player statistics for each game."""
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
    
    # Fantasy Points
    fantasy_points_standard: Mapped[float] = mapped_column(Float, default=0.0)
    fantasy_points_ppr: Mapped[float] = mapped_column(Float, default=0.0)
    fantasy_points_half_ppr: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Metadata
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


class BettingLine(Base):
    """Betting lines and props for games and players."""
    __tablename__ = 'betting_lines'
    
    line_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Game Reference
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey('games.game_id'), nullable=False)
    
    # Line Type
    line_type: Mapped[str] = mapped_column(String(50), nullable=False)  # spread/total/moneyline/player_prop
    
    # Player Props (nullable for game lines)
    player_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey('players.player_id'))
    prop_type: Mapped[Optional[str]] = mapped_column(String(50))  # passing_yards/rushing_yards/etc
    
    # Line Values
    line_value: Mapped[float] = mapped_column(Float, nullable=False)
    over_odds: Mapped[Optional[int]] = mapped_column(Integer)  # American odds
    under_odds: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Game Lines
    home_spread: Mapped[Optional[float]] = mapped_column(Float)
    away_spread: Mapped[Optional[float]] = mapped_column(Float)
    total_points: Mapped[Optional[float]] = mapped_column(Float)
    home_moneyline: Mapped[Optional[int]] = mapped_column(Integer)
    away_moneyline: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Source and Timing
    sportsbook: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    game = relationship("Game", back_populates="betting_lines")
    
    __table_args__ = (
        Index('idx_betting_game_type', 'game_id', 'line_type'),
        Index('idx_betting_player_prop', 'player_id', 'prop_type'),
        Index('idx_betting_timestamp', 'timestamp'),
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
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # ensemble/xgb/neural/etc
    
    # Predictions
    predicted_passing_yards: Mapped[Optional[float]] = mapped_column(Float)
    predicted_passing_tds: Mapped[Optional[float]] = mapped_column(Float)
    predicted_rushing_yards: Mapped[Optional[float]] = mapped_column(Float)
    predicted_rushing_tds: Mapped[Optional[float]] = mapped_column(Float)
    predicted_receiving_yards: Mapped[Optional[float]] = mapped_column(Float)
    predicted_receiving_tds: Mapped[Optional[float]] = mapped_column(Float)
    predicted_receptions: Mapped[Optional[float]] = mapped_column(Float)
    predicted_fantasy_points: Mapped[Optional[float]] = mapped_column(Float)
    
    # Confidence Scores (0-1)
    confidence_overall: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_passing: Mapped[Optional[float]] = mapped_column(Float)
    confidence_rushing: Mapped[Optional[float]] = mapped_column(Float)
    confidence_receiving: Mapped[Optional[float]] = mapped_column(Float)
    
    # Feature Importance (JSON)
    feature_importance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Metadata
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_player_game', 'player_id', 'game_id'),
        Index('idx_prediction_model', 'model_version', 'model_type'),
        UniqueConstraint('player_id', 'game_id', 'model_version'),
    )


class GamePrediction(Base):
    """ML model predictions for game outcomes."""
    __tablename__ = 'game_predictions'
    
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Reference
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey('games.game_id'), nullable=False)
    
    # Model Info
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Predictions
    predicted_home_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_away_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_total_points: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_spread: Mapped[float] = mapped_column(Float, nullable=False)  # Positive = home favorite
    
    # Win Probabilities
    home_win_probability: Mapped[float] = mapped_column(Float, nullable=False)
    away_win_probability: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Confidence and Edge
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    betting_edge: Mapped[Optional[float]] = mapped_column(Float)  # vs market lines
    
    # Metadata
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    game = relationship("Game", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_game_prediction_game', 'game_id'),
        Index('idx_game_prediction_model', 'model_version', 'model_type'),
        UniqueConstraint('game_id', 'model_version'),
    )


class FeatureStore(Base):
    """Engineered features for ML models."""
    __tablename__ = 'feature_store'
    
    feature_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Reference
    player_id: Mapped[str] = mapped_column(String(50), ForeignKey('players.player_id'), nullable=False)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey('games.game_id'), nullable=False)
    
    # Feature Categories
    recent_form_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # Last N games stats
    seasonal_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)     # Season averages
    opponent_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)     # Opponent-adjusted
    contextual_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)   # Weather, venue, etc
    advanced_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)     # Complex engineered
    
    # Feature Version
    feature_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_feature_player_game', 'player_id', 'game_id'),
        Index('idx_feature_version', 'feature_version'),
        UniqueConstraint('player_id', 'game_id', 'feature_version'),
    )


class ModelPerformance(Base):
    """Track model performance and metrics."""
    __tablename__ = 'model_performance'
    
    performance_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Model Info
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    evaluation_period: Mapped[str] = mapped_column(String(50), nullable=False)  # 2023_season/week_1/etc
    
    # Performance Metrics
    accuracy_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    betting_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    feature_importance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Evaluation Details
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    evaluation_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_model_performance_version', 'model_version'),
        Index('idx_model_performance_period', 'evaluation_period'),
    )


# Database Initialization Functions
def create_all_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all tables from the database."""
    Base.metadata.drop_all(engine)


# Example usage and testing
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create test database
    engine = create_engine("sqlite:///nfl_predictions.db", echo=True)
    create_all_tables(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Add sample data
    sample_player = Player(
        player_id="mahomes_patrick",
        name="Patrick Mahomes",
        first_name="Patrick",
        last_name="Mahomes",
        position="QB",
        current_team="KC",
        height_inches=75,
        weight_lbs=230,
        age=28
    )
    
    session.add(sample_player)
    session.commit()
    session.close()
    
    print("Database schema created successfully!")