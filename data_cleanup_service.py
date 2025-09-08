"""
Data Cleanup Service - Fix player activity status and data quality
"""

import logging
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from simplified_database_models import Player, PlayerGameStats
from datetime import datetime, date
import requests

logger = logging.getLogger(__name__)

class DataCleanupService:
    def __init__(self, session: Session):
        self.session = session
        
    def update_player_activity_status(self):
        """Update player activity status based on 2024 season data"""
        
        # List of definitively retired players
        retired_players = [
            'Tom Brady', 'Matt Ryan', 'Ben Roethlisberger', 'Russell Wilson', 
            'Drew Brees', 'Philip Rivers', 'Cam Newton', 'Andy Dalton',
            'Joe Flacco', 'Ryan Fitzpatrick', 'Blake Bortles', 'Case Keenum'
        ]
        
        retired_count = 0
        # Mark retired players as inactive
        for name in retired_players:
            players = self.session.query(Player).filter(
                Player.name.ilike(f"%{name}%")
            ).all()
            
            for player in players:
                if player.is_active:
                    player.is_active = False
                    player.updated_at = datetime.now()
                    retired_count += 1
                    logger.info(f"Marked {player.name} as retired")
        
        # Mark players without recent stats as inactive
        cutoff_date = date(2024, 1, 1)
        
        # Get players who have no stats since 2024
        inactive_query = text("""
            UPDATE players 
            SET is_active = 0, updated_at = :now
            WHERE player_id NOT IN (
                SELECT DISTINCT player_id 
                FROM player_game_stats 
                WHERE created_at >= :cutoff_date
            )
            AND is_active = 1
        """)
        
        result = self.session.execute(inactive_query, {
            'now': datetime.now(),
            'cutoff_date': cutoff_date
        })
        
        inactive_count = result.rowcount
        
        self.session.commit()
        
        print(f"✅ Updated player activity status:")
        print(f"   - {retired_count} retired players marked inactive")
        print(f"   - {inactive_count} players without 2024 stats marked inactive")
        
        return retired_count + inactive_count
    
    def validate_data_quality(self):
        """Validate and report data quality issues"""
        
        # Check for duplicate players
        duplicate_query = text("""
            SELECT name, COUNT(*) as count
            FROM players
            WHERE is_active = 1
            GROUP BY name
            HAVING COUNT(*) > 1
        """)
        
        duplicates = self.session.execute(duplicate_query).fetchall()
        
        # Check for players with no stats
        no_stats_query = text("""
            SELECT COUNT(*)
            FROM players p
            LEFT JOIN player_game_stats pgs ON p.player_id = pgs.player_id
            WHERE p.is_active = 1 AND pgs.player_id IS NULL
        """)
        
        no_stats_count = self.session.execute(no_stats_query).scalar()
        
        # Check for active players with old data only
        old_data_query = text("""
            SELECT COUNT(DISTINCT p.player_id)
            FROM players p
            JOIN player_game_stats pgs ON p.player_id = pgs.player_id
            WHERE p.is_active = 1 
            AND pgs.created_at < :cutoff_date
            AND p.player_id NOT IN (
                SELECT DISTINCT player_id 
                FROM player_game_stats 
                WHERE created_at >= :cutoff_date
            )
        """)
        
        old_data_count = self.session.execute(old_data_query, {
            'cutoff_date': date(2024, 1, 1)
        }).scalar()
        
        print(f"\n📊 DATA QUALITY REPORT:")
        print(f"   - Duplicate active players: {len(duplicates)}")
        print(f"   - Active players with no stats: {no_stats_count}")
        print(f"   - Active players with only old data: {old_data_count}")
        
        if duplicates:
            print("\n🔍 DUPLICATE PLAYERS FOUND:")
            for name, count in duplicates:
                print(f"   - {name}: {count} entries")
        
        return {
            'duplicates': len(duplicates),
            'no_stats': no_stats_count,
            'old_data': old_data_count
        }
    
    def clean_duplicate_players(self):
        """Remove duplicate player entries, keeping the most recent"""
        
        # Find and merge duplicate players
        duplicate_query = text("""
            SELECT name, GROUP_CONCAT(player_id) as player_ids, COUNT(*) as count
            FROM players
            WHERE is_active = 1
            GROUP BY name
            HAVING COUNT(*) > 1
        """)
        
        duplicates = self.session.execute(duplicate_query).fetchall()
        cleaned_count = 0
        
        for name, player_ids_str, count in duplicates:
            player_ids = player_ids_str.split(',')
            
            # Keep the player with the most recent data
            best_player_query = text("""
                SELECT p.player_id, MAX(pgs.created_at) as latest_stat
                FROM players p
                LEFT JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                WHERE p.player_id IN :player_ids
                GROUP BY p.player_id
                ORDER BY latest_stat DESC NULLS LAST
                LIMIT 1
            """)
            
            best_player = self.session.execute(best_player_query, {
                'player_ids': tuple(player_ids)
            }).fetchone()
            
            if best_player:
                keep_id = best_player[0]
                
                # Mark other duplicates as inactive
                for player_id in player_ids:
                    if player_id != keep_id:
                        update_query = text("""
                            UPDATE players 
                            SET is_active = 0, updated_at = :now
                            WHERE player_id = :player_id
                        """)
                        
                        self.session.execute(update_query, {
                            'now': datetime.now(),
                            'player_id': player_id
                        })
                        cleaned_count += 1
        
        self.session.commit()
        print(f"✅ Cleaned {cleaned_count} duplicate player entries")
        return cleaned_count
    
    def update_current_teams(self):
        """Update current team information for active players"""
        
        try:
            import nfl_data_py as nfl
            
            # Get current roster data
            current_rosters = nfl.import_rosters([2024])
            
            updated_count = 0
            
            for _, roster_player in current_rosters.iterrows():
                player = self.session.query(Player).filter(
                    Player.name == roster_player['full_name']
                ).first()
                
                if player and player.is_active:
                    if player.current_team != roster_player['team']:
                        player.current_team = roster_player['team']
                        player.updated_at = datetime.now()
                        updated_count += 1
            
            self.session.commit()
            print(f"✅ Updated {updated_count} player team assignments")
            return updated_count
            
        except Exception as e:
            print(f"⚠️  Could not update team assignments: {e}")
            return 0
    
    def full_cleanup(self):
        """Run complete data cleanup process"""
        
        print("🧹 Starting comprehensive data cleanup...")
        print("=" * 50)
        
        # Step 1: Update activity status
        activity_updates = self.update_player_activity_status()
        
        # Step 2: Validate data quality
        quality_report = self.validate_data_quality()
        
        # Step 3: Clean duplicates
        duplicate_cleanups = self.clean_duplicate_players()
        
        # Step 4: Update team assignments
        team_updates = self.update_current_teams()
        
        print(f"\n🎉 CLEANUP COMPLETE!")
        print(f"   - Total player updates: {activity_updates}")
        print(f"   - Duplicates cleaned: {duplicate_cleanups}")
        print(f"   - Team updates: {team_updates}")
        
        return {
            'activity_updates': activity_updates,
            'duplicate_cleanups': duplicate_cleanups,
            'team_updates': team_updates,
            'quality_report': quality_report
        }
