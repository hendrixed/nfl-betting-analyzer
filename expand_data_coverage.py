"""
Data Coverage Expander

This module dramatically expands the data coverage to include
all active NFL players and recent historical data for comprehensive predictions.
"""

import logging
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from database_models import Player, PlayerGameStats, Base
from enhanced_data_collector import EnhancedNFLDataCollector
from historical_data_standardizer import HistoricalDataStandardizer

logger = logging.getLogger(__name__)

class DataCoverageExpander:
    """Expand data coverage to include all active NFL players"""
    
    def __init__(self, db_path: str = "nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Current data stats
        self.current_players = 0
        self.current_records = 0
        
        # Target data stats  
        self.target_players = 1500  # Aim for ~1500 NFL players (active + recent)
        self.target_records = 100000  # Aim for ~100K statistical records
        
    def analyze_current_coverage(self) -> Dict[str, any]:
        """Analyze current data coverage gaps"""
        
        logger.info("📊 Analyzing current data coverage...")
        
        try:
            # Count current data using raw SQL to avoid schema issues
            players_query = text("SELECT COUNT(*) FROM players")
            players_result = self.session.execute(players_query).fetchone()
            self.current_players = players_result[0] if players_result else 0
            
            records_query = text("SELECT COUNT(*) FROM player_game_stats")
            records_result = self.session.execute(records_query).fetchone()
            self.current_records = records_result[0] if records_result else 0
            
            # Analyze by position
            position_query = text("""
                SELECT 
                    p.position,
                    COUNT(DISTINCT p.player_id) as player_count,
                    COUNT(pgs.stat_id) as record_count,
                    AVG(pgs.fantasy_points_ppr) as avg_fantasy_points
                FROM players p
                LEFT JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
                GROUP BY p.position
            """)
            
            position_results = self.session.execute(position_query).fetchall()
            
            coverage_analysis = {
                'total_players': self.current_players,
                'total_records': self.current_records,
                'coverage_percentage': (self.current_players / self.target_players) * 100,
                'records_percentage': (self.current_records / self.target_records) * 100,
                'position_breakdown': {}
            }
            
            for row in position_results:
                position = row[0]
                coverage_analysis['position_breakdown'][position] = {
                    'players': int(row[1]),
                    'records': int(row[2]),
                    'avg_fantasy': float(row[3] or 0)
                }
            
            # Identify gaps
            coverage_analysis['critical_gaps'] = []
            coverage_analysis['expansion_needs'] = []
            
            if self.current_players < 50:
                coverage_analysis['critical_gaps'].append("Severely limited player coverage")
            elif self.current_players < 200:
                coverage_analysis['critical_gaps'].append("Limited player coverage")
            
            if self.current_records < 5000:
                coverage_analysis['critical_gaps'].append("Insufficient historical data")
            elif self.current_records < 25000:
                coverage_analysis['expansion_needs'].append("Expand historical data")
            
            logger.info(f"Current coverage: {self.current_players} players, {self.current_records} records")
            logger.info(f"Target coverage: {self.target_players} players, {self.target_records} records")
            
            return coverage_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing coverage: {e}")
            return {}
    
    def expand_player_coverage(self) -> bool:
        """Expand player coverage to include all recent NFL players"""
        
        logger.info("🔄 Expanding player coverage...")
        
        try:
            # Get comprehensive roster data for recent seasons
            expansion_seasons = [2022, 2023, 2024]  # Expand to include more recent seasons
            
            all_players_added = 0
            
            for season in expansion_seasons:
                logger.info(f"   Processing {season} season rosters...")
                
                try:
                    # Get roster data
                    rosters = nfl.import_seasonal_rosters([season])
                    
                    if rosters.empty:
                        logger.warning(f"No roster data for {season}")
                        continue
                    
                    # Filter to skill positions only
                    skill_rosters = rosters[rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
                    
                    season_players_added = 0
                    
                    for _, row in skill_rosters.iterrows():
                        try:
                            # Create player ID (consistent with enhanced system)
                            player_id = row.get('player_id')
                            if not player_id:
                                continue
                            
                            # Check if player already exists
                            existing_query = text("SELECT COUNT(*) FROM players WHERE player_id = :player_id")
                            existing_result = self.session.execute(existing_query, {"player_id": player_id}).fetchone()
                            
                            if existing_result[0] == 0:
                                # Create new player using raw SQL to avoid schema issues
                                insert_query = text("""
                                    INSERT INTO players (
                                        player_id, name, first_name, last_name, position, 
                                        current_team, height_inches, weight_lbs, draft_year, 
                                        college, is_active, created_at, updated_at
                                    ) VALUES (
                                        :player_id, :name, :first_name, :last_name, :position,
                                        :current_team, :height_inches, :weight_lbs, :draft_year,
                                        :college, :is_active, :created_at, :updated_at
                                    )
                                """)
                                
                                self.session.execute(insert_query, {
                                    "player_id": player_id,
                                    "name": row.get('player_name', ''),
                                    "first_name": row.get('first_name', ''),
                                    "last_name": row.get('last_name', ''),
                                    "position": row.get('position', ''),
                                    "current_team": row.get('team', ''),
                                    "height_inches": self._convert_height(row.get('height')),
                                    "weight_lbs": int(row.get('weight', 0)) if pd.notna(row.get('weight')) else None,
                                    "draft_year": int(row.get('entry_year', 0)) if pd.notna(row.get('entry_year')) else None,
                                    "college": row.get('college', ''),
                                    "is_active": row.get('status', '').upper() in ['ACT', 'RES'],
                                    "created_at": datetime.now(),
                                    "updated_at": datetime.now()
                                })
                                
                                season_players_added += 1
                                all_players_added += 1
                                
                        except Exception as e:
                            logger.warning(f"Error processing player {row.get('player_name', 'unknown')}: {e}")
                            continue
                    
                    self.session.commit()
                    logger.info(f"   ✅ {season}: Added {season_players_added} new players")
                    
                except Exception as e:
                    logger.error(f"Error processing {season} rosters: {e}")
                    continue
            
            logger.info(f"✅ Player expansion completed: {all_players_added} new players added")
            return all_players_added > 0
            
        except Exception as e:
            logger.error(f"Error expanding player coverage: {e}")
            return False
    
    def expand_statistical_coverage(self) -> bool:
        """Expand statistical data coverage"""
        
        logger.info("📈 Expanding statistical coverage...")
        
        try:
            # Get comprehensive weekly data for recent seasons
            expansion_seasons = [2022, 2023, 2024]
            
            all_records_added = 0
            
            for season in expansion_seasons:
                logger.info(f"   Processing {season} season statistics...")
                
                try:
                    # Get weekly stats
                    weekly_stats = nfl.import_weekly_data([season])
                    
                    if weekly_stats.empty:
                        logger.warning(f"No weekly stats for {season}")
                        continue
                    
                    # Get existing player IDs using raw SQL
                    existing_players_query = text("SELECT player_id FROM players")
                    existing_players_result = self.session.execute(existing_players_query).fetchall()
                    existing_player_ids = set([row[0] for row in existing_players_result])
                    
                    # Filter weekly stats to existing players
                    player_stats = weekly_stats[weekly_stats['player_id'].isin(existing_player_ids)].copy()
                    
                    season_records_added = 0
                    
                    for _, row in player_stats.iterrows():
                        try:
                            # Create game ID
                            week = int(row.get('week', 0))
                            team = row.get('recent_team', '')
                            opponent = row.get('opponent_team', '')
                            game_id = f"{season}_{week:02d}_{team}_{opponent}"
                            
                            # Check if record exists
                            existing_query = text("""
                                SELECT COUNT(*) FROM player_game_stats 
                                WHERE player_id = :player_id AND game_id = :game_id
                            """)
                            existing_result = self.session.execute(existing_query, {
                                "player_id": row.get('player_id'),
                                "game_id": game_id
                            }).fetchone()
                            
                            if existing_result[0] == 0:
                                # Create new stat record using raw SQL with all required fields
                                insert_query = text("""
                                    INSERT INTO player_game_stats (
                                        player_id, game_id, team, opponent, is_home,
                                        passing_attempts, passing_completions, passing_yards, 
                                        passing_touchdowns, passing_interceptions, passing_sacks, passing_sack_yards,
                                        rushing_attempts, rushing_yards, rushing_touchdowns, rushing_fumbles, rushing_first_downs,
                                        targets, receptions, receiving_yards, receiving_touchdowns, receiving_fumbles, receiving_first_downs,
                                        snap_count, snap_percentage, routes_run, air_yards, yards_after_catch,
                                        fantasy_points_standard, fantasy_points_ppr, fantasy_points_half_ppr,
                                        created_at, updated_at
                                    ) VALUES (
                                        :player_id, :game_id, :team, :opponent, :is_home,
                                        :passing_attempts, :passing_completions, :passing_yards,
                                        :passing_touchdowns, :passing_interceptions, :passing_sacks, :passing_sack_yards,
                                        :rushing_attempts, :rushing_yards, :rushing_touchdowns, :rushing_fumbles, :rushing_first_downs,
                                        :targets, :receptions, :receiving_yards, :receiving_touchdowns, :receiving_fumbles, :receiving_first_downs,
                                        :snap_count, :snap_percentage, :routes_run, :air_yards, :yards_after_catch,
                                        :fantasy_points_standard, :fantasy_points_ppr, :fantasy_points_half_ppr,
                                        :created_at, :updated_at
                                    )
                                """)
                                
                                self.session.execute(insert_query, {
                                    "player_id": row.get('player_id'),
                                    "game_id": game_id,
                                    "team": team,
                                    "opponent": opponent,
                                    "is_home": row.get('is_home', False),
                                    "passing_attempts": int(row.get('attempts', 0)),
                                    "passing_completions": int(row.get('completions', 0)),
                                    "passing_yards": int(row.get('passing_yards', 0)),
                                    "passing_touchdowns": int(row.get('passing_tds', 0)),
                                    "passing_interceptions": int(row.get('interceptions', 0)),
                                    "passing_sacks": int(row.get('sacks', 0)),
                                    "passing_sack_yards": int(row.get('sack_yards', 0)),
                                    "rushing_attempts": int(row.get('carries', 0)),
                                    "rushing_yards": int(row.get('rushing_yards', 0)),
                                    "rushing_touchdowns": int(row.get('rushing_tds', 0)),
                                    "rushing_fumbles": int(row.get('rushing_fumbles', 0)),
                                    "rushing_first_downs": int(row.get('rushing_first_downs', 0)),
                                    "targets": int(row.get('targets', 0)),
                                    "receptions": int(row.get('receptions', 0)),
                                    "receiving_yards": int(row.get('receiving_yards', 0)),
                                    "receiving_touchdowns": int(row.get('receiving_tds', 0)),
                                    "receiving_fumbles": int(row.get('receiving_fumbles', 0)),
                                    "receiving_first_downs": int(row.get('receiving_first_downs', 0)),
                                    "snap_count": int(row.get('snap_count', 0)),
                                    "snap_percentage": float(row.get('snap_percentage', 0)),
                                    "routes_run": int(row.get('routes_run', 0)),
                                    "air_yards": int(row.get('air_yards', 0)),
                                    "yards_after_catch": int(row.get('yards_after_catch', 0)),
                                    "fantasy_points_standard": float(row.get('fantasy_points', 0)),
                                    "fantasy_points_ppr": float(row.get('fantasy_points_ppr', 0)),
                                    "fantasy_points_half_ppr": float(row.get('fantasy_points_half_ppr', 0)),
                                    "created_at": datetime.now(),
                                    "updated_at": datetime.now()
                                })
                                
                                season_records_added += 1
                                all_records_added += 1
                                
                        except Exception as e:
                            logger.warning(f"Error processing stat record: {e}")
                            continue
                    
                    # Commit in batches
                    if season_records_added > 0:
                        self.session.commit()
                        logger.info(f"   ✅ {season}: Added {season_records_added} new stat records")
                    
                except Exception as e:
                    logger.error(f"Error processing {season} stats: {e}")
                    continue
            
            logger.info(f"✅ Statistical expansion completed: {all_records_added} new records added")
            return all_records_added > 0
            
        except Exception as e:
            logger.error(f"Error expanding statistical coverage: {e}")
            return False
    
    def _convert_height(self, height_str: str) -> Optional[int]:
        """Convert height string to inches"""
        if not height_str or pd.isna(height_str):
            return None
        
        try:
            # Handle format like "6-2" or "6'2"
            height_str = str(height_str).replace("'", "-").replace('"', '')
            if '-' in height_str:
                feet, inches = height_str.split('-')
                return int(feet) * 12 + int(inches)
            else:
                return int(height_str)
        except:
            return None
    
    def run_complete_expansion(self) -> bool:
        """Run complete data coverage expansion"""
        
        logger.info("🚀 Starting complete data coverage expansion...")
        
        try:
            # Step 1: Analyze current coverage
            coverage = self.analyze_current_coverage()
            
            logger.info(f"📊 Current: {coverage.get('total_players', 0)} players, {coverage.get('total_records', 0)} records")
            logger.info(f"📈 Target: {self.target_players} players, {self.target_records} records")
            
            # Step 2: Expand player coverage
            players_success = self.expand_player_coverage()
            
            # Step 3: Expand statistical coverage  
            stats_success = self.expand_statistical_coverage()
            
            # Step 4: Final analysis
            final_coverage = self.analyze_current_coverage()
            
            final_players = final_coverage.get('total_players', 0)
            final_records = final_coverage.get('total_records', 0)
            
            logger.info(f"📊 Final: {final_players} players, {final_records} records")
            
            success = (final_players > self.current_players * 2 and 
                      final_records > self.current_records * 2)
            
            if success:
                logger.info("🎉 Data coverage expansion completed successfully!")
                return True
            else:
                logger.warning("⚠️ Data coverage expansion had limited impact")
                return False
            
        except Exception as e:
            logger.error(f"❌ Data coverage expansion failed: {e}")
            return False

def main():
    """Run data coverage expansion"""
    
    print("📈 NFL Data Coverage Expander")
    print("=" * 50)
    
    expander = DataCoverageExpander()
    success = expander.run_complete_expansion()
    
    if success:
        print("\n✅ Data coverage expansion COMPLETED successfully!")
        print("   The database now has comprehensive player and statistical data.")
        return True
    else:
        print("\n❌ Data coverage expansion FAILED!")
        print("   Manual data collection may be required.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
