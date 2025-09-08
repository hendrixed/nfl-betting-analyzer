"""
Simplified Database Models for Comprehensive Stats Engine
Matches the actual database schema without the enhanced fields
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Player(Base):
    """Simplified player model matching actual database"""
    __tablename__ = 'players'
    
    player_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    position = Column(String(10), nullable=False)
    current_team = Column(String(10))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    game_stats = relationship("PlayerGameStats", back_populates="player")

class PlayerGameStats(Base):
    """Simplified game stats model matching actual database"""
    __tablename__ = 'player_game_stats'
    
    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey('players.player_id'), nullable=False)
    game_id = Column(String(50), nullable=False)
    team = Column(String(10), nullable=False)
    opponent = Column(String(10), nullable=False)
    is_home = Column(Boolean, nullable=False)
    
    # Passing Stats
    passing_attempts = Column(Integer, default=0)
    passing_completions = Column(Integer, default=0)
    passing_yards = Column(Integer, default=0)
    passing_touchdowns = Column(Integer, default=0)
    passing_interceptions = Column(Integer, default=0)
    passing_sacks = Column(Integer, default=0)
    passing_sack_yards = Column(Integer, default=0)
    
    # Rushing Stats
    rushing_attempts = Column(Integer, default=0)
    rushing_yards = Column(Integer, default=0)
    rushing_touchdowns = Column(Integer, default=0)
    rushing_fumbles = Column(Integer, default=0)
    rushing_first_downs = Column(Integer, default=0)
    
    # Receiving Stats
    targets = Column(Integer, default=0)
    receptions = Column(Integer, default=0)
    receiving_yards = Column(Integer, default=0)
    receiving_touchdowns = Column(Integer, default=0)
    receiving_fumbles = Column(Integer, default=0)
    receiving_first_downs = Column(Integer, default=0)
    
    # Advanced Stats
    snap_count = Column(Integer)
    snap_percentage = Column(Float)
    routes_run = Column(Integer)
    air_yards = Column(Integer)
    yards_after_catch = Column(Integer)
    
    # Fantasy Points
    fantasy_points_standard = Column(Float, default=0.0)
    fantasy_points_ppr = Column(Float, default=0.0)
    fantasy_points_half_ppr = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    player = relationship("Player", back_populates="game_stats")

class Game(Base):
    """Simplified game model"""
    __tablename__ = 'games'
    
    game_id = Column(String(50), primary_key=True)
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    home_team = Column(String(10), nullable=False)
    away_team = Column(String(10), nullable=False)
    game_date = Column(DateTime)
    
def get_db():
    """Database session dependency"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///nfl_predictions.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
