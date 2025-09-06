#!/usr/bin/env python3
"""
Diagnose Data Mismatch Between Tables
Check why player_game_stats and players tables aren't joining properly.
"""

import sqlite3
from pathlib import Path

def diagnose_data_mismatch():
    """Diagnose the data mismatch between tables."""
    
    db_path = Path("data/nfl_predictions.db")
    
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔍 DIAGNOSING DATA MISMATCH")
    print("=" * 50)
    
    # Check players table content
    print("\n📊 PLAYERS TABLE ANALYSIS:")
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    print(f"Total players: {player_count}")
    
    if player_count > 0:
        cursor.execute("SELECT player_id, name, position FROM players LIMIT 10")
        players = cursor.fetchall()
        print("Sample players:")
        for player in players:
            print(f"  - {player[0]} | {player[1]} | {player[2]}")
    
    # Check player_game_stats table content
    print("\n📈 PLAYER_GAME_STATS TABLE ANALYSIS:")
    cursor.execute("SELECT COUNT(*) FROM player_game_stats")
    stats_count = cursor.fetchone()[0]
    print(f"Total game stats: {stats_count}")
    
    if stats_count > 0:
        cursor.execute("SELECT DISTINCT player_id FROM player_game_stats LIMIT 10")
        stats_players = cursor.fetchall()
        print("Sample player_ids from game stats:")
        for player in stats_players:
            print(f"  - {player[0]}")
    
    # Check for any matches
    print("\n🔗 JOIN ANALYSIS:")
    cursor.execute("""
        SELECT COUNT(*) FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
    """)
    join_count = cursor.fetchone()[0]
    print(f"Records that join successfully: {join_count}")
    
    # Check unique player_ids in each table
    cursor.execute("SELECT COUNT(DISTINCT player_id) FROM players")
    unique_players = cursor.fetchone()[0]
    print(f"Unique player_ids in players table: {unique_players}")
    
    cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_game_stats") 
    unique_stats_players = cursor.fetchone()[0]
    print(f"Unique player_ids in player_game_stats table: {unique_stats_players}")
    
    # Find mismatches
    print("\n🕵️ MISMATCH INVESTIGATION:")
    
    # Players in stats but not in players table
    cursor.execute("""
        SELECT DISTINCT pgs.player_id 
        FROM player_game_stats pgs
        LEFT JOIN players p ON pgs.player_id = p.player_id
        WHERE p.player_id IS NULL
        LIMIT 10
    """)
    orphaned_stats = cursor.fetchall()
    
    if orphaned_stats:
        print(f"Found {len(orphaned_stats)} player_ids in game_stats but not in players table:")
        for player in orphaned_stats:
            print(f"  - {player[0]}")
    else:
        print("✅ All game_stats player_ids exist in players table")
    
    # Sample game stats with all info
    print("\n📋 SAMPLE GAME STATS (with null positions):")
    cursor.execute("""
        SELECT pgs.player_id, pgs.passing_yards, pgs.rushing_yards, pgs.receiving_yards,
               pgs.fantasy_points_standard, p.position
        FROM player_game_stats pgs
        LEFT JOIN players p ON pgs.player_id = p.player_id
        WHERE pgs.fantasy_points_standard > 0
        LIMIT 5
    """)
    sample_stats = cursor.fetchall()
    
    for stat in sample_stats:
        position = stat[5] if stat[5] else "UNKNOWN"
        print(f"  - {stat[0]}: {stat[1]}py, {stat[2]}ry, {stat[3]}rec, {stat[4]}fp | Position: {position}")
    
    conn.close()
    
    # Provide solution
    print("\n💡 SOLUTION:")
    if join_count == 0:
        print("❌ No successful JOINs - player_id formats don't match")
        print("🔧 Need to either:")
        print("   1. Fix the player_id format mismatch")
        print("   2. Create a version that works without the players table")
        print("   3. Re-populate the players table with correct player_ids")
    else:
        print(f"✅ {join_count} successful JOINs found")
        print("🔧 Should work with some data")

if __name__ == "__main__":
    diagnose_data_mismatch()