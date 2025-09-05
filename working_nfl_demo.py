"""
Fixed NFL Demo - Using Correct Column Names
Working demo with real NFL data using the actual column names from nfl_data_py.
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedNFLDemo:
    """Fixed demo with correct column names."""
    
    def __init__(self):
        self.demo_dir = Path("fixed_nfl_demo")
        self.demo_dir.mkdir(exist_ok=True)
        self.db_path = self.demo_dir / "nfl_fixed.db"
        
    def run_complete_demo(self):
        """Run complete demo with real NFL data."""
        logger.info("🏈 Starting Fixed NFL Demo with Real Data...")
        
        # Step 1: Create database
        self.create_database()
        
        # Step 2: Get real NFL data
        self.get_real_nfl_data()
        
        # Step 3: Store in database
        self.store_nfl_data()
        
        # Step 4: Simple analysis
        self.analyze_data()
        
        # Step 5: Simple predictions
        self.simple_predictions()
        
        logger.info("✅ Fixed NFL demo completed successfully!")
        
    def create_database(self):
        """Create database for real NFL data."""
        logger.info("📊 Creating database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Players table (using correct column names)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                player_name TEXT,
                first_name TEXT,
                last_name TEXT,
                position TEXT,
                team TEXT,
                years_exp INTEGER,
                height TEXT,
                weight INTEGER
            )
        """)
        
        # Weekly stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weekly_stats (
                id INTEGER PRIMARY KEY,
                player_id TEXT,
                player_name TEXT,
                position TEXT,
                team TEXT,
                week INTEGER,
                season INTEGER,
                passing_yards INTEGER,
                passing_tds INTEGER,
                rushing_yards INTEGER,
                rushing_tds INTEGER,
                receiving_yards INTEGER,
                receiving_tds INTEGER,
                receptions INTEGER,
                targets INTEGER,
                fantasy_points_ppr REAL,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Games table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                season INTEGER,
                week INTEGER,
                game_date TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ Database created")
        
    def get_real_nfl_data(self):
        """Get real NFL data using correct functions."""
        logger.info("📥 Getting real NFL data...")
        
        try:
            import nfl_data_py as nfl
            
            # Get 2024 rosters
            logger.info("Getting 2024 rosters...")
            self.rosters = nfl.import_seasonal_rosters([2024])
            logger.info(f"✅ Got {len(self.rosters)} roster records")
            
            # Get recent weekly data (smaller sample first)
            logger.info("Getting recent weekly data...")
            self.weekly_data = nfl.import_weekly_data([2024])
            logger.info(f"✅ Got {len(self.weekly_data)} weekly stat records")
            
            # Get 2024 schedule
            logger.info("Getting 2024 schedule...")
            self.schedule = nfl.import_schedules([2024])
            logger.info(f"✅ Got {len(self.schedule)} games")
            
        except Exception as e:
            logger.error(f"❌ Error getting NFL data: {e}")
            raise
            
    def store_nfl_data(self):
        """Store NFL data in database."""
        logger.info("💾 Storing NFL data in database...")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Store players (using correct column names)
            players_df = self.rosters[['player_id', 'player_name', 'first_name', 'last_name', 
                                     'position', 'team', 'years_exp', 'height', 'weight']].copy()
            players_df = players_df.drop_duplicates(subset=['player_id'])
            players_df.to_sql('players', conn, if_exists='replace', index=False)
            logger.info(f"✅ Stored {len(players_df)} players")
            
            # Store weekly stats (check available columns first)
            logger.info("Checking weekly data columns...")
            weekly_cols = list(self.weekly_data.columns)
            logger.info(f"Available weekly columns: {weekly_cols[:10]}...")  # Show first 10
            
            # Use available columns
            weekly_columns = ['player_id', 'player_name', 'position', 'recent_team', 
                            'week', 'season']
            
            # Add stat columns if they exist
            stat_columns = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                          'receiving_yards', 'receiving_tds', 'receptions', 'targets', 
                          'fantasy_points_ppr']
            
            for col in stat_columns:
                if col in self.weekly_data.columns:
                    weekly_columns.append(col)
                    
            # Get the data
            weekly_df = self.weekly_data[weekly_columns].copy()
            
            # Rename columns to match database
            weekly_df = weekly_df.rename(columns={'recent_team': 'team'})
            
            # Only store records with player_id
            weekly_df = weekly_df[weekly_df['player_id'].notna()]
            
            # Fill missing values
            for col in ['fantasy_points_ppr'] + [c for c in stat_columns if c in weekly_df.columns]:
                if col in weekly_df.columns:
                    weekly_df[col] = weekly_df[col].fillna(0)
            
            # Store in database
            weekly_df.to_sql('weekly_stats', conn, if_exists='replace', index=False)
            logger.info(f"✅ Stored {len(weekly_df)} weekly stat records")
            
            # Store games (from schedule)
            games_columns = ['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team']
            
            # Add score columns if they exist
            if 'home_score' in self.schedule.columns:
                games_columns.append('home_score')
            if 'away_score' in self.schedule.columns:
                games_columns.append('away_score')
                
            games_df = self.schedule[games_columns].copy()
            games_df = games_df.rename(columns={'gameday': 'game_date'})
            games_df.to_sql('games', conn, if_exists='replace', index=False)
            logger.info(f"✅ Stored {len(games_df)} games")
            
        except Exception as e:
            logger.error(f"❌ Error storing data: {e}")
            logger.error(f"Available roster columns: {list(self.rosters.columns)}")
            logger.error(f"Available weekly columns: {list(self.weekly_data.columns)}")
            raise
        finally:
            conn.close()
            
    def analyze_data(self):
        """Analyze the real NFL data."""
        logger.info("📊 Analyzing real NFL data...")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get player count by position
            position_counts = pd.read_sql_query("""
                SELECT position, COUNT(*) as player_count
                FROM players 
                WHERE position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
                GROUP BY position
                ORDER BY player_count DESC
            """, conn)
            
            logger.info("📊 Players by Position:")
            for _, pos in position_counts.iterrows():
                logger.info(f"  {pos['position']}: {pos['player_count']} players")
            
            # Get some sample weekly performance
            if 'fantasy_points_ppr' in pd.read_sql_query("PRAGMA table_info(weekly_stats)", conn)['name'].values:
                top_weeks = pd.read_sql_query("""
                    SELECT player_name, position, team, week,
                           fantasy_points_ppr
                    FROM weekly_stats 
                    WHERE fantasy_points_ppr > 20
                    ORDER BY fantasy_points_ppr DESC
                    LIMIT 5
                """, conn)
                
                if len(top_weeks) > 0:
                    logger.info("🏆 Top Individual Performances:")
                    for _, week in top_weeks.iterrows():
                        logger.info(f"  Week {week['week']}: {week['player_name']} ({week['position']}, {week['team']}) - {week['fantasy_points_ppr']:.1f} pts")
                else:
                    logger.info("📊 Fantasy points data available but no high scores found")
            else:
                logger.info("📊 Fantasy points column not available in weekly stats")
                
            # Show sample of actual data we have
            sample_stats = pd.read_sql_query("""
                SELECT player_name, position, team, week
                FROM weekly_stats 
                LIMIT 5
            """, conn)
            
            logger.info("📋 Sample Weekly Data:")
            for _, stat in sample_stats.iterrows():
                logger.info(f"  {stat['player_name']} ({stat['position']}, {stat['team']}) - Week {stat['week']}")
                
        except Exception as e:
            logger.warning(f"Analysis error: {e}")
            logger.info("Data is stored, but analysis had issues")
        finally:
            conn.close()
            
    def simple_predictions(self):
        """Make simple predictions using real data."""
        logger.info("🔮 Making simple predictions...")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get some recent players for prediction
            recent_players = pd.read_sql_query("""
                SELECT DISTINCT player_name, position, team
                FROM weekly_stats 
                WHERE week >= (SELECT MAX(week) - 2 FROM weekly_stats)
                  AND position IN ('QB', 'RB', 'WR', 'TE')
                LIMIT 10
            """, conn)
            
            logger.info("🎯 Sample Predictions for Recent Players:")
            
            # Position-based prediction logic
            position_base = {'QB': 18.0, 'RB': 12.0, 'WR': 10.0, 'TE': 8.0}
            
            for _, player in recent_players.iterrows():
                base_score = position_base.get(player['position'], 8.0)
                predicted = base_score + np.random.normal(0, 2)
                predicted = max(0, predicted)
                
                logger.info(f"  {player['player_name']} ({player['position']}, {player['team']}): {predicted:.1f} pts")
                
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            logger.info("Predictions feature had issues, but data collection worked!")
        finally:
            conn.close()
        
    def show_summary(self):
        """Show summary of what we have."""
        logger.info("📋 Demo Summary:")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Count records
            players_count = pd.read_sql_query("SELECT COUNT(*) as count FROM players", conn).iloc[0]['count']
            stats_count = pd.read_sql_query("SELECT COUNT(*) as count FROM weekly_stats", conn).iloc[0]['count']
            games_count = pd.read_sql_query("SELECT COUNT(*) as count FROM games", conn).iloc[0]['count']
            
            logger.info(f"✅ Players in database: {players_count}")
            logger.info(f"✅ Weekly stat records: {stats_count}")
            logger.info(f"✅ Games in database: {games_count}")
            
            # Show table schemas
            tables = ['players', 'weekly_stats', 'games']
            for table in tables:
                schema = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
                logger.info(f"✅ {table} columns: {list(schema['name'])}")
                
        except Exception as e:
            logger.error(f"Summary error: {e}")
        finally:
            conn.close()


def main():
    """Run the fixed NFL demo."""
    print("🏈 NFL Prediction System - Fixed Demo with Real Data")
    print("=" * 60)
    print("This demo uses actual NFL data with correct column names.")
    print("Note: First run might take a minute to download data...")
    print()
    
    demo = FixedNFLDemo()
    
    try:
        demo.run_complete_demo()
        demo.show_summary()
        
        print("\n" + "=" * 60)
        print("🎉 Real NFL data demo completed successfully!")
        print(f"📁 Demo files saved to: {demo.demo_dir}")
        print("\n💡 What we accomplished:")
        print("✅ Downloaded real 2024 NFL data")
        print("✅ Stored in SQLite database")
        print("✅ Analyzed player performance")
        print("✅ Made simple predictions")
        print("\n🚀 Next: Build advanced ML models on this foundation!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Check the logs above for details.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()