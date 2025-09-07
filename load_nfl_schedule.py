"""
NFL Schedule Loader

This module loads current and upcoming NFL game schedules
to enable prediction testing and real-time functionality.
"""

import logging
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

class NFLScheduleLoader:
    """Load NFL game schedules for predictions"""
    
    def __init__(self, db_path: str = "nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        self.current_season = 2024
        self.schedules_loaded = 0
        
    def create_schedule_table(self) -> bool:
        """Create schedule table if it doesn't exist"""
        
        logger.info("📅 Creating schedule table...")
        
        try:
            create_table_query = text("""
                CREATE TABLE IF NOT EXISTS nfl_schedule (
                    game_id TEXT PRIMARY KEY,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    game_type TEXT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    game_date TEXT,
                    game_time TEXT,
                    stadium TEXT,
                    weather_temperature REAL,
                    weather_wind_mph REAL,
                    weather_humidity REAL,
                    home_score INTEGER,
                    away_score INTEGER,
                    total_score INTEGER,
                    spread_line REAL,
                    total_line REAL,
                    home_moneyline INTEGER,
                    away_moneyline INTEGER,
                    is_completed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.session.execute(create_table_query)
            self.session.commit()
            
            logger.info("✅ Schedule table created/verified")
            return True
            
        except Exception as e:
            logger.error(f"Error creating schedule table: {e}")
            return False
    
    def load_season_schedule(self, season: int = 2024) -> bool:
        """Load complete season schedule"""
        
        logger.info(f"📅 Loading {season} season schedule...")
        
        try:
            # Get schedule data
            schedule_data = nfl.import_schedules([season])
            
            if schedule_data.empty:
                logger.warning(f"No schedule data available for {season}")
                return False
            
            games_loaded = 0
            
            for _, game in schedule_data.iterrows():
                try:
                    # Create game ID
                    week = int(game.get('week', 0))
                    home_team = game.get('home_team', '')
                    away_team = game.get('away_team', '')
                    game_id = f"{season}_{week:02d}_{away_team}_{home_team}"
                    
                    # Check if game already exists
                    existing_query = text("SELECT COUNT(*) FROM nfl_schedule WHERE game_id = :game_id")
                    existing_result = self.session.execute(existing_query, {"game_id": game_id}).fetchone()
                    
                    if existing_result[0] == 0:
                        # Insert new game
                        insert_query = text("""
                            INSERT INTO nfl_schedule (
                                game_id, season, week, game_type, home_team, away_team,
                                game_date, game_time, stadium, home_score, away_score,
                                total_score, spread_line, total_line, home_moneyline, away_moneyline,
                                is_completed, created_at, updated_at
                            ) VALUES (
                                :game_id, :season, :week, :game_type, :home_team, :away_team,
                                :game_date, :game_time, :stadium, :home_score, :away_score,
                                :total_score, :spread_line, :total_line, :home_moneyline, :away_moneyline,
                                :is_completed, :created_at, :updated_at
                            )
                        """)
                        
                        # Parse game date
                        game_date = game.get('gameday')
                        if pd.notna(game_date):
                            game_date = str(game_date)
                        else:
                            game_date = None
                        
                        # Determine if game is completed
                        home_score = game.get('home_score')
                        away_score = game.get('away_score')
                        is_completed = pd.notna(home_score) and pd.notna(away_score)
                        
                        self.session.execute(insert_query, {
                            "game_id": game_id,
                            "season": season,
                            "week": week,
                            "game_type": game.get('game_type', 'REG'),
                            "home_team": home_team,
                            "away_team": away_team,
                            "game_date": game_date,
                            "game_time": game.get('gametime'),
                            "stadium": game.get('stadium'),
                            "home_score": int(home_score) if pd.notna(home_score) else None,
                            "away_score": int(away_score) if pd.notna(away_score) else None,
                            "total_score": int(home_score + away_score) if is_completed else None,
                            "spread_line": float(game.get('spread_line', 0)) if pd.notna(game.get('spread_line')) else None,
                            "total_line": float(game.get('total_line', 0)) if pd.notna(game.get('total_line')) else None,
                            "home_moneyline": int(game.get('home_moneyline', 0)) if pd.notna(game.get('home_moneyline')) else None,
                            "away_moneyline": int(game.get('away_moneyline', 0)) if pd.notna(game.get('away_moneyline')) else None,
                            "is_completed": is_completed,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        })
                        
                        games_loaded += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing game: {e}")
                    continue
            
            self.session.commit()
            self.schedules_loaded += games_loaded
            
            logger.info(f"✅ {season}: Loaded {games_loaded} games")
            return games_loaded > 0
            
        except Exception as e:
            logger.error(f"Error loading {season} schedule: {e}")
            return False
    
    def get_upcoming_games(self, weeks_ahead: int = 2) -> List[Dict]:
        """Get upcoming games for prediction testing"""
        
        logger.info(f"🔍 Finding upcoming games ({weeks_ahead} weeks ahead)...")
        
        try:
            # Get current week (approximate)
            current_date = datetime.now()
            
            # Query for upcoming games
            upcoming_query = text("""
                SELECT game_id, season, week, home_team, away_team, game_date, is_completed
                FROM nfl_schedule 
                WHERE is_completed = FALSE 
                AND season = :season
                ORDER BY week ASC, game_date ASC
                LIMIT 20
            """)
            
            results = self.session.execute(upcoming_query, {"season": self.current_season}).fetchall()
            
            upcoming_games = []
            for row in results:
                upcoming_games.append({
                    'game_id': row[0],
                    'season': row[1], 
                    'week': row[2],
                    'home_team': row[3],
                    'away_team': row[4],
                    'game_date': row[5],
                    'is_completed': row[6]
                })
            
            logger.info(f"✅ Found {len(upcoming_games)} upcoming games")
            return upcoming_games
            
        except Exception as e:
            logger.error(f"Error getting upcoming games: {e}")
            return []
    
    def analyze_schedule_coverage(self) -> Dict[str, any]:
        """Analyze schedule data coverage"""
        
        logger.info("📊 Analyzing schedule coverage...")
        
        try:
            # Count total games
            total_query = text("SELECT COUNT(*) FROM nfl_schedule")
            total_result = self.session.execute(total_query).fetchone()
            total_games = total_result[0] if total_result else 0
            
            # Count by completion status
            completed_query = text("SELECT COUNT(*) FROM nfl_schedule WHERE is_completed = TRUE")
            completed_result = self.session.execute(completed_query).fetchone()
            completed_games = completed_result[0] if completed_result else 0
            
            upcoming_games = total_games - completed_games
            
            # Count by week
            week_query = text("""
                SELECT week, COUNT(*) as game_count, 
                       SUM(CASE WHEN is_completed THEN 1 ELSE 0 END) as completed_count
                FROM nfl_schedule 
                WHERE season = :season
                GROUP BY week 
                ORDER BY week
            """)
            
            week_results = self.session.execute(week_query, {"season": self.current_season}).fetchall()
            
            week_breakdown = {}
            for row in week_results:
                week_breakdown[int(row[0])] = {
                    'total': int(row[1]),
                    'completed': int(row[2]),
                    'upcoming': int(row[1]) - int(row[2])
                }
            
            coverage_analysis = {
                'total_games': total_games,
                'completed_games': completed_games,
                'upcoming_games': upcoming_games,
                'completion_percentage': (completed_games / total_games * 100) if total_games > 0 else 0,
                'week_breakdown': week_breakdown,
                'current_season': self.current_season
            }
            
            logger.info(f"📊 Schedule: {total_games} total, {completed_games} completed, {upcoming_games} upcoming")
            
            return coverage_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing schedule coverage: {e}")
            return {}
    
    def run_complete_schedule_load(self) -> bool:
        """Run complete schedule loading process"""
        
        logger.info("🚀 Starting complete NFL schedule loading...")
        
        try:
            # Step 1: Create schedule table
            table_success = self.create_schedule_table()
            if not table_success:
                return False
            
            # Step 2: Load recent seasons
            seasons_to_load = [2022, 2023, 2024]
            
            total_success = True
            for season in seasons_to_load:
                season_success = self.load_season_schedule(season)
                if not season_success:
                    logger.warning(f"Failed to load {season} schedule")
                    total_success = False
            
            # Step 3: Analyze coverage
            coverage = self.analyze_schedule_coverage()
            
            # Step 4: Get upcoming games for testing
            upcoming = self.get_upcoming_games()
            
            success = (self.schedules_loaded > 100 and len(upcoming) > 0)
            
            if success:
                logger.info("🎉 NFL schedule loading completed successfully!")
                logger.info(f"   Loaded {self.schedules_loaded} games")
                logger.info(f"   Found {len(upcoming)} upcoming games for predictions")
                return True
            else:
                logger.warning("⚠️ NFL schedule loading had limited success")
                return False
            
        except Exception as e:
            logger.error(f"❌ NFL schedule loading failed: {e}")
            return False

def main():
    """Run NFL schedule loading"""
    
    print("📅 NFL Schedule Loader")
    print("=" * 50)
    
    loader = NFLScheduleLoader()
    success = loader.run_complete_schedule_load()
    
    if success:
        print("\n✅ NFL schedule loading COMPLETED successfully!")
        print("   The database now has current NFL game schedules.")
        return True
    else:
        print("\n❌ NFL schedule loading FAILED!")
        print("   Manual schedule loading may be required.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
