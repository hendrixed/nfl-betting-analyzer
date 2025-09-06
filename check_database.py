#!/usr/bin/env python3
"""
Check Database Contents and Configuration
Diagnose where the NFL data was actually stored.
"""

import sys
from pathlib import Path
import os
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import sqlite3

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import get_config

def check_sqlite_database():
    """Check if SQLite database exists and has data."""
    sqlite_path = Path("data/nfl_predictions.db")
    
    print(f"🔍 Checking SQLite database: {sqlite_path}")
    
    if not sqlite_path.exists():
        print(f"❌ SQLite database not found at {sqlite_path}")
        return False
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"📊 Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check player_game_stats specifically
        if ('player_game_stats',) in tables:
            cursor.execute("SELECT COUNT(*) FROM player_game_stats;")
            count = cursor.fetchone()[0]
            print(f"📈 player_game_stats table has {count} records")
            
            if count > 0:
                cursor.execute("""
                    SELECT pgs.player_id, p.position 
                    FROM player_game_stats pgs
                    JOIN players p ON pgs.player_id = p.player_id 
                    LIMIT 5
                """)
                samples = cursor.fetchall()
                print("📋 Sample records:")
                for sample in samples:
                    print(f"  - {sample[0]} ({sample[1]})")
        else:
            print("❌ player_game_stats table not found")
            
        # Check players table
        if ('players',) in tables:
            cursor.execute("SELECT COUNT(*) FROM players;")
            count = cursor.fetchone()[0]
            print(f"👥 players table has {count} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error checking SQLite database: {e}")
        return False

def check_postgresql_database():
    """Check if PostgreSQL database exists and has data."""
    print(f"\n🔍 Checking PostgreSQL database from config...")
    
    try:
        # Get config
        config = get_config()
        database_url = config.database.url
        
        print(f"📍 Database URL: {database_url}")
        
        if not database_url or "postgresql" not in database_url:
            print("❌ No PostgreSQL database configured")
            return False
        
        # Try to connect
        engine = create_engine(database_url)
        
        # Check if database is accessible
        with engine.connect() as conn:
            # Check tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            print(f"📊 Found {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
            
            # Check player_game_stats specifically
            if 'player_game_stats' in tables:
                result = conn.execute(text("SELECT COUNT(*) FROM player_game_stats;"))
                count = result.scalar()
                print(f"📈 player_game_stats table has {count} records")
                
                if count > 0:
                    result = conn.execute(text("""
                        SELECT pgs.player_id, p.position 
                        FROM player_game_stats pgs
                        JOIN players p ON pgs.player_id = p.player_id 
                        LIMIT 5
                    """))
                    samples = result.fetchall()
                    print("📋 Sample records:")
                    for sample in samples:
                        print(f"  - {sample[0]} ({sample[1]})")
                    return True
            else:
                print("❌ player_game_stats table not found")
                
            # Check players table
            if 'players' in tables:
                result = conn.execute(text("SELECT COUNT(*) FROM players;"))
                count = result.scalar()
                print(f"👥 players table has {count} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking PostgreSQL database: {e}")
        return False

def check_config_issues():
    """Check configuration issues."""
    print(f"\n🔧 Checking configuration...")
    
    try:
        config = get_config()
        
        print(f"📍 Database type: {config.database.type}")
        print(f"📍 Database URL: {config.database.url}")
        print(f"📍 SQLite path: {config.database.sqlite_path}")
        
        # Check if config file exists
        config_path = Path("config/config.yaml")
        if config_path.exists():
            print(f"⚙️ Config file exists: {config_path}")
            with open(config_path, 'r') as f:
                content = f.read()
                print("📄 Config file content:")
                print(content[:500] + "..." if len(content) > 500 else content)
        else:
            print(f"❌ Config file not found: {config_path}")
            
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")

def main():
    """Main function to diagnose database issues."""
    print("🏈 NFL DATABASE DIAGNOSTICS")
    print("=" * 50)
    
    # Check configuration first
    check_config_issues()
    
    # Check SQLite database
    sqlite_has_data = check_sqlite_database()
    
    # Check PostgreSQL database  
    postgres_has_data = check_postgresql_database()
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 20)
    
    if sqlite_has_data:
        print("✅ SQLite database has data - betting predictor should work")
        print("💡 Solution: Your current setup should work!")
        
    elif postgres_has_data:
        print("✅ PostgreSQL database has data")
        print("❌ But betting predictor is looking in SQLite")
        print("💡 Solution: Update betting predictor to use PostgreSQL")
        
    else:
        print("❌ No data found in either database")
        print("💡 Solution: Run data collection again or use sample data")
    
    print(f"\n🚀 NEXT STEPS:")
    if postgres_has_data and not sqlite_has_data:
        print("1. Run: python fix_database_config.py")
        print("2. Or run: python unified_betting_predictor.py --use-config-db")
    elif not sqlite_has_data and not postgres_has_data:
        print("1. Run: python populate_sample_data.py")
        print("2. Or re-run: python run_nfl_system.py download-data --seasons 2023,2024")
    else:
        print("1. Run: python unified_betting_predictor.py")

if __name__ == "__main__":
    main()