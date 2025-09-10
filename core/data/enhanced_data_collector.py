"""
Enhanced NFL Data Collector - New Hierarchical Data Collection Pipeline

This module implements the new data collection system that uses authoritative
NFL data sources and proper validation to ensure accurate starter identification
and eliminate incorrect backup player statistics.
"""

import logging
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from core.data.data_foundation import (
    MasterPlayer, WeeklyRosterSnapshot, PlayerRole, ValidationReport,
    PlayerGameValidation, TeamDepthChart, DataSourceMetrics
)

logger = logging.getLogger(__name__)


class EnhancedNFLDataCollector:
    """Enhanced data collector using hierarchical data architecture"""
    
    def __init__(self, session: Session, current_season: int = 2025):
        self.session = session
        self.current_season = current_season
        self.data_cache = {}
        
        # Data source reliability tracking
        self.source_metrics = {
            'rosters': DataSourceMetrics('NFL_Rosters'),
            'depth_charts': DataSourceMetrics('NFL_DepthCharts'),
            'snap_counts': DataSourceMetrics('NFL_SnapCounts'),
            'injuries': DataSourceMetrics('NFL_Injuries'),
            'weekly_stats': DataSourceMetrics('NFL_WeeklyStats')
        }
        
    async def collect_weekly_foundation_data(self, week: int) -> Dict[str, WeeklyRosterSnapshot]:
        """
        CRITICAL: Collect authoritative data for all teams for a specific week
        Returns: Dict[team_code, WeeklyRosterSnapshot]
        """
        logger.info(f"Collecting foundation data for {self.current_season} Week {week}")
        
        try:
            # Step 1: Get Official Rosters (MOST AUTHORITATIVE)
            rosters = self._get_official_rosters()
            if rosters.empty:
                logger.error("Failed to load roster data - cannot proceed")
                return {}
            
            # Step 2: Get Depth Charts
            depth_charts = self._get_depth_charts()
            
            # Step 3: Get Snap Counts (Actual Usage)
            snap_counts = self._get_snap_counts(week)
            
            # Step 4: Get Injury Reports
            injuries = self._get_injury_reports()
            
            # Step 5: Create Master Player Records with Cross-Validation
            team_snapshots = {}
            
            teams = rosters['team'].dropna().unique()
            logger.info(f"Processing {len(teams)} teams")
            
            for team in teams:
                if pd.isna(team) or not team or team == '':
                    continue
                    
                try:
                    team_snapshot = self._create_team_roster_snapshot(
                        team=team,
                        week=week, 
                        rosters=rosters,
                        depth_charts=depth_charts,
                        snap_counts=snap_counts,
                        injuries=injuries
                    )
                    
                    if team_snapshot:
                        team_snapshots[team] = team_snapshot
                        logger.debug(f"Created snapshot for {team}: {len(team_snapshot.get_active_players())} active players")
                    
                except Exception as e:
                    logger.error(f"Failed to create snapshot for team {team}: {e}")
                    continue
                
            logger.info(f"Created roster snapshots for {len(team_snapshots)} teams")
            return team_snapshots
            
        except Exception as e:
            logger.error(f"Critical error in collect_weekly_foundation_data: {e}")
            return {}
    
    def _get_official_rosters(self) -> pd.DataFrame:
        """Get official NFL rosters - MOST AUTHORITATIVE SOURCE"""
        try:
            logger.info(f"Loading roster data for {self.current_season}")
            
            # CRITICAL: Use current season only
            rosters = nfl.import_seasonal_rosters([self.current_season])
            
            if rosters.empty:
                logger.error(f"No roster data available for {self.current_season}")
                return pd.DataFrame()
            
            # Validate required columns exist
            required_cols = ['player_id', 'player_name', 'position', 'team', 'status']
            available_cols = rosters.columns.tolist()
            missing_cols = [col for col in required_cols if col not in available_cols]
            
            if missing_cols:
                logger.error(f"Missing required columns in roster data: {missing_cols}")
                logger.info(f"Available columns: {available_cols}")
                
            # Filter to skill positions only
            skill_positions = ['QB', 'RB', 'WR', 'TE', 'K']
            rosters = rosters[rosters['position'].isin(skill_positions)].copy()
            
            # Clean data
            rosters = rosters.dropna(subset=['player_id', 'player_name', 'position', 'team'])
            
            # Update source metrics
            self.source_metrics['rosters'].players_covered = len(rosters)
            self.source_metrics['rosters'].teams_covered = rosters['team'].nunique()
            self.source_metrics['rosters'].positions_covered = rosters['position'].unique().tolist()
            self.source_metrics['rosters'].last_assessment = datetime.now()
            
            logger.info(f"Loaded {len(rosters)} roster records for {self.current_season}")
            return rosters
            
        except Exception as e:
            logger.error(f"Failed to load roster data: {e}")
            self.source_metrics['rosters'].uptime_percentage = 0.0
            return pd.DataFrame()
    
    def _get_depth_charts(self) -> pd.DataFrame:
        """Get official depth charts"""
        try:
            logger.info(f"Loading depth chart data for {self.current_season}")
            depth_charts = nfl.import_depth_charts([self.current_season])
            
            if not depth_charts.empty:
                # Filter to skill positions
                skill_positions = ['QB', 'RB', 'WR', 'TE', 'K']
                if 'pos_abb' in depth_charts.columns:
                    depth_charts = depth_charts[depth_charts['pos_abb'].isin(skill_positions)].copy()
                
                self.source_metrics['depth_charts'].players_covered = len(depth_charts)
                self.source_metrics['depth_charts'].teams_covered = depth_charts['team'].nunique() if 'team' in depth_charts.columns else 0
                
            logger.info(f"Loaded depth chart data for {len(depth_charts)} players")
            return depth_charts
            
        except Exception as e:
            logger.error(f"Failed to load depth chart data: {e}")
            self.source_metrics['depth_charts'].uptime_percentage = 0.0
            return pd.DataFrame()
    
    def _get_snap_counts(self, week: int) -> pd.DataFrame:
        """Get snap count data for recent weeks"""
        try:
            logger.info(f"Loading snap count data for {self.current_season}")
            
            # Get snap counts for the season
            snap_counts = nfl.import_snap_counts([self.current_season])
            
            if not snap_counts.empty:
                # Filter to relevant weeks only (current and previous weeks)
                recent_weeks = snap_counts[snap_counts['week'] <= week].copy()
                
                # Filter to skill positions
                skill_positions = ['QB', 'RB', 'WR', 'TE', 'K']
                if 'position' in recent_weeks.columns:
                    recent_weeks = recent_weeks[recent_weeks['position'].isin(skill_positions)].copy()
                
                self.source_metrics['snap_counts'].players_covered = recent_weeks['player_id'].nunique() if 'player_id' in recent_weeks.columns else 0
                
                logger.info(f"Loaded snap count data for {len(recent_weeks)} player-games")
                return recent_weeks
            else:
                logger.warning(f"No snap count data available for {self.current_season}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load snap count data: {e}")
            self.source_metrics['snap_counts'].uptime_percentage = 0.0
            return pd.DataFrame()
    
    def _get_injury_reports(self) -> pd.DataFrame:
        """Get current injury reports"""
        try:
            logger.info(f"Loading injury data for {self.current_season}")
            injuries = nfl.import_injuries([self.current_season])
            
            if not injuries.empty:
                # Filter to current/recent reports only
                if 'report_date' in injuries.columns:
                    recent_injuries = injuries[
                        pd.to_datetime(injuries['report_date']) >= (datetime.now() - timedelta(days=7))
                    ].copy()
                else:
                    recent_injuries = injuries.copy()
                
                self.source_metrics['injuries'].players_covered = len(recent_injuries)
                
                logger.info(f"Loaded {len(recent_injuries)} recent injury reports")
                return recent_injuries
            else:
                logger.warning(f"No injury data available for {self.current_season}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load injury data: {e}")
            self.source_metrics['injuries'].uptime_percentage = 0.0
            return pd.DataFrame()
    
    def _create_team_roster_snapshot(self, team: str, week: int,
                                   rosters: pd.DataFrame, 
                                   depth_charts: pd.DataFrame,
                                   snap_counts: pd.DataFrame,
                                   injuries: pd.DataFrame) -> Optional[WeeklyRosterSnapshot]:
        """Create validated roster snapshot for a team"""
        
        try:
            # Get team data
            team_roster = rosters[rosters['team'] == team].copy()
            
            if team_roster.empty:
                logger.warning(f"No roster data found for team {team}")
                return None
            
            team_depth = depth_charts[depth_charts['team'] == team].copy() if 'team' in depth_charts.columns else pd.DataFrame()
            team_snaps = snap_counts[snap_counts['team'] == team].copy() if 'team' in snap_counts.columns else pd.DataFrame()
            team_injuries = injuries[injuries['team'] == team].copy() if 'team' in injuries.columns else pd.DataFrame()
            
            # Create master player records
            master_players = []
            
            for _, roster_row in team_roster.iterrows():
                try:
                    master_player = self._create_master_player(
                        roster_row=roster_row,
                        depth_charts=team_depth,
                        snap_counts=team_snaps,
                        injuries=team_injuries
                    )
                    
                    if master_player:
                        master_players.append(master_player)
                        
                except Exception as e:
                    logger.warning(f"Failed to create master player for {roster_row.get('player_name', 'unknown')}: {e}")
                    continue
            
            if not master_players:
                logger.warning(f"No valid players created for team {team}")
                return None
            
            # Classify players by role
            starters = [p for p in master_players if p.role_classification == PlayerRole.STARTER]
            backup_primary = [p for p in master_players if p.role_classification == PlayerRole.BACKUP_HIGH]
            backup_depth = [p for p in master_players if p.role_classification == PlayerRole.BACKUP_LOW]
            inactive = [p for p in master_players if p.role_classification == PlayerRole.INACTIVE]
            
            return WeeklyRosterSnapshot(
                season=self.current_season,
                week=week,
                team=team,
                snapshot_date=datetime.now(),
                starters=starters,
                backup_primary=backup_primary,
                backup_depth=backup_depth,
                inactive=inactive,
                depth_chart_confidence=self._calculate_depth_chart_confidence(team_depth, team_snaps),
                injury_impact_score=self._calculate_injury_impact(team_injuries, master_players)
            )
            
        except Exception as e:
            logger.error(f"Error creating team roster snapshot for {team}: {e}")
            return None
    
    def _create_master_player(self, roster_row: pd.Series,
                            depth_charts: pd.DataFrame,
                            snap_counts: pd.DataFrame, 
                            injuries: pd.DataFrame) -> Optional[MasterPlayer]:
        """Create a master player record with cross-validation"""
        
        try:
            # CRITICAL: Use official player ID as primary key
            player_id = roster_row.get('player_id')
            if not player_id or pd.isna(player_id):
                logger.warning(f"No player_id for {roster_row.get('player_name', 'unknown')}")
                return None
            
            # Get player name and position
            name = roster_row.get('player_name', '')
            position = roster_row.get('position', '')
            
            if not name or not position:
                logger.warning(f"Missing name or position for player_id {player_id}")
                return None
            
            # Only process skill position players
            if position not in ['QB', 'RB', 'WR', 'TE', 'K']:
                return None
            
            # Get depth chart info
            depth_rank = self._get_depth_chart_rank(player_id, name, position, depth_charts)
            
            # Get recent snap usage
            avg_snap_rate = self._calculate_avg_snap_rate(player_id, snap_counts)
            
            # Check injury status
            is_injured = self._check_injury_status(player_id, name, injuries)
            
            # Classify role
            role = self._classify_player_role(
                depth_rank=depth_rank,
                snap_rate=avg_snap_rate,
                status=roster_row.get('status', ''),
                is_injured=is_injured,
                position=position
            )
            
            return MasterPlayer(
                nfl_id=str(player_id),
                gsis_id=roster_row.get('gsis_id'),
                pfr_id=roster_row.get('pfr_id'),
                espn_id=roster_row.get('espn_id'),
                name=name,
                first_name=roster_row.get('first_name', ''),
                last_name=roster_row.get('last_name', ''),
                position=position,
                current_team=roster_row.get('team', ''),
                roster_status=roster_row.get('status', ''),
                depth_chart_rank=depth_rank,
                role_classification=role,
                avg_snap_rate_3_games=avg_snap_rate,
                is_injured=is_injured,
                data_quality_score=self._calculate_data_quality_score(roster_row, depth_rank, avg_snap_rate),
                last_validated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating master player: {e}")
            return None
    
    def _get_depth_chart_rank(self, player_id: str, name: str, position: str, depth_charts: pd.DataFrame) -> Optional[int]:
        """Get player's depth chart ranking"""
        if depth_charts.empty:
            return None
        
        # Try to match by player_id first
        if 'player_id' in depth_charts.columns:
            player_depth = depth_charts[depth_charts['player_id'] == player_id]
            if not player_depth.empty and 'pos_rank' in player_depth.columns:
                return int(player_depth['pos_rank'].iloc[0])
        
        # Fallback to name and position matching
        if 'player_name' in depth_charts.columns and 'pos_abb' in depth_charts.columns:
            last_name = name.split()[-1] if name else ''
            player_depth = depth_charts[
                (depth_charts['player_name'].str.contains(last_name, case=False, na=False)) &
                (depth_charts['pos_abb'] == position)
            ]
            
            if not player_depth.empty and 'pos_rank' in player_depth.columns:
                return int(player_depth['pos_rank'].iloc[0])
        
        return None
    
    def _classify_player_role(self, depth_rank: Optional[int], 
                            snap_rate: float, 
                            status: str, 
                            is_injured: bool,
                            position: str) -> PlayerRole:
        """
        CRITICAL: Classify player role based on multiple authoritative signals
        This determines whether player should have stats tracked
        """
        
        # Inactive players
        if status.upper() not in ['ACT', 'RES'] or is_injured:
            return PlayerRole.INACTIVE
        
        # Position-specific logic
        if position == 'K':
            # Kickers are usually starters if active
            return PlayerRole.STARTER if snap_rate > 0.5 or depth_rank == 1 else PlayerRole.BACKUP_LOW
        
        # No depth chart data - use snap rate only
        if depth_rank is None:
            if snap_rate > 0.7:
                return PlayerRole.STARTER
            elif snap_rate > 0.3:
                return PlayerRole.BACKUP_HIGH
            elif snap_rate > 0.1:
                return PlayerRole.BACKUP_LOW
            else:
                return PlayerRole.INACTIVE
        
        # Use both depth chart and snap rate
        if depth_rank == 1:
            # Listed as starter
            if snap_rate > 0.5:  # Actually playing starter snaps
                return PlayerRole.STARTER
            elif snap_rate > 0.2:  # Some starter missed time
                return PlayerRole.BACKUP_HIGH
            else:  # Listed as starter but not playing
                return PlayerRole.BACKUP_LOW
                
        elif depth_rank == 2:
            # Listed as primary backup
            if snap_rate > 0.4:  # Playing significant snaps
                return PlayerRole.BACKUP_HIGH
            elif snap_rate > 0.1:  # Getting some work
                return PlayerRole.BACKUP_LOW
            else:
                return PlayerRole.INACTIVE
                
        else:
            # Depth chart rank 3+
            if snap_rate > 0.3:  # Unexpectedly high usage
                return PlayerRole.BACKUP_HIGH
            elif snap_rate > 0.1:
                return PlayerRole.BACKUP_LOW
            else:
                return PlayerRole.INACTIVE
    
    def _calculate_avg_snap_rate(self, player_id: str, snap_counts: pd.DataFrame) -> float:
        """Calculate average snap rate over recent games"""
        if snap_counts.empty:
            return 0.0
        
        # Get player's snap data
        if 'player_id' in snap_counts.columns:
            player_snaps = snap_counts[snap_counts['player_id'] == player_id].copy()
        else:
            return 0.0
        
        if player_snaps.empty:
            return 0.0
        
        # Get last 3 games
        recent_snaps = player_snaps.tail(3)
        
        if 'snap_pct' in recent_snaps.columns:
            snap_rates = recent_snaps['snap_pct'].dropna()
        elif 'snaps' in recent_snaps.columns and 'team_snaps' in recent_snaps.columns:
            # Calculate snap percentage
            team_snaps = recent_snaps['team_snaps'].replace(0, 1)  # Avoid division by zero
            snap_rates = (recent_snaps['snaps'] / team_snaps).dropna()
        else:
            return 0.0
        
        return float(snap_rates.mean()) if len(snap_rates) > 0 else 0.0
    
    def _check_injury_status(self, player_id: str, name: str, injuries: pd.DataFrame) -> bool:
        """Check if player is currently injured"""
        if injuries.empty:
            return False
        
        # Check by player_id first
        if 'player_id' in injuries.columns:
            player_injuries = injuries[injuries['player_id'] == player_id]
            if not player_injuries.empty and 'status' in player_injuries.columns:
                return any(status in ['OUT', 'DOUBTFUL'] for status in player_injuries['status'])
        
        # Fallback to name matching
        if 'player_name' in injuries.columns:
            name_injuries = injuries[injuries['player_name'].str.contains(name, case=False, na=False)]
            if not name_injuries.empty and 'status' in name_injuries.columns:
                return any(status in ['OUT', 'DOUBTFUL'] for status in name_injuries['status'])
        
        return False
    
    def _calculate_data_quality_score(self, roster_row: pd.Series, 
                                    depth_rank: Optional[int], 
                                    snap_rate: float) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        
        # Has basic roster info
        if roster_row.get('player_id') and roster_row.get('player_name'):
            score += 0.4
        
        # Has depth chart data
        if depth_rank is not None:
            score += 0.3
        
        # Has recent snap data
        if snap_rate > 0:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_depth_chart_confidence(self, depth_charts: pd.DataFrame, snap_counts: pd.DataFrame) -> float:
        """Calculate how reliable the depth chart is"""
        if depth_charts.empty:
            return 0.5  # Medium confidence when no depth chart data
        
        # Simple confidence based on data availability
        if snap_counts.empty:
            return 0.7  # Decent confidence with depth chart but no validation
        
        return 0.8  # Good confidence with both depth chart and snap data
    
    def _calculate_injury_impact(self, injuries: pd.DataFrame, players: List[MasterPlayer]) -> float:
        """Calculate how much injuries have impacted the depth chart"""
        if not players:
            return 0.0
        
        injured_count = sum(1 for p in players if p.is_injured)
        total_count = len(players)
        
        return injured_count / total_count if total_count > 0 else 0.0


class RoleBasedStatsCollector:
    """Collect stats only for players who actually played meaningful snaps"""
    
    def __init__(self, enhanced_collector: EnhancedNFLDataCollector):
        self.enhanced_collector = enhanced_collector
        
    async def collect_validated_stats(self, week: int, 
                                    team_snapshots: Dict[str, WeeklyRosterSnapshot]) -> List[Dict]:
        """
        CRITICAL: Only collect stats for players who should have them
        """
        logger.info(f"Collecting validated stats for week {week}")
        
        # Get raw weekly stats
        try:
            raw_stats = nfl.import_weekly_data([self.enhanced_collector.current_season])
            week_stats = raw_stats[raw_stats['week'] == week].copy()
            logger.info(f"Loaded {len(week_stats)} raw stat records for week {week}")
        except Exception as e:
            logger.error(f"Failed to load weekly stats: {e}")
            return []
        
        # Get snap counts for validation
        try:
            snap_counts = nfl.import_snap_counts([self.enhanced_collector.current_season])
            week_snaps = snap_counts[snap_counts['week'] == week].copy()
            logger.info(f"Loaded {len(week_snaps)} snap count records for validation")
        except Exception as e:
            logger.error(f"Failed to load snap counts: {e}")
            week_snaps = pd.DataFrame()
        
        validated_stats = []
        excluded_count = 0
        
        for _, stat_row in week_stats.iterrows():
            try:
                # Find player in roster snapshots
                player_found = False
                team = stat_row.get('recent_team', '')
                
                if team in team_snapshots:
                    snapshot = team_snapshots[team]
                    stat_eligible_players = snapshot.get_stat_eligible_players()
                    
                    # Look for player by ID
                    player_id = str(stat_row.get('player_id', ''))
                    matching_player = None
                    
                    for player in stat_eligible_players:
                        if player.nfl_id == player_id:
                            matching_player = player
                            player_found = True
                            break
                    
                    if player_found and matching_player:
                        # Validate stats against snap data
                        if self._validate_stats_against_snaps(stat_row, week_snaps, matching_player):
                            validated_stat = self._create_validated_stat_record(stat_row, matching_player)
                            validated_stats.append(validated_stat)
                        else:
                            logger.warning(f"Invalid stats for {player_id}: stats without sufficient snaps")
                            excluded_count += 1
                    else:
                        # Player not in eligible list - check if they should be excluded
                        if self._has_significant_stats(stat_row):
                            logger.warning(f"Player {player_id} has stats but not in eligible list")
                        excluded_count += 1
                
                if not player_found:
                    logger.debug(f"Player {stat_row.get('player_id', 'unknown')} not found in eligible players")
                    excluded_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing stats for {stat_row.get('player_id', 'unknown')}: {e}")
                excluded_count += 1
                continue
        
        logger.info(f"Validated {len(validated_stats)} stat records from {len(week_stats)} raw records")
        logger.info(f"Excluded {excluded_count} records due to validation failures")
        return validated_stats
    
    def _validate_stats_against_snaps(self, stats: pd.Series, 
                                    snaps: pd.DataFrame, 
                                    player: MasterPlayer) -> bool:
        """Ensure player actually played before attributing stats"""
        
        # Inactive players should have minimal/no stats
        if player.role_classification == PlayerRole.INACTIVE:
            return not self._has_significant_stats(stats)
        
        # Players with significant stats should have snaps
        if self._has_significant_stats(stats):
            if snaps.empty:
                # No snap data available - allow stats for starters and primary backups
                return player.role_classification in [PlayerRole.STARTER, PlayerRole.BACKUP_HIGH]
            
            player_snaps = snaps[snaps['player_id'] == player.nfl_id]
            if player_snaps.empty:
                # No snaps recorded but has stats - suspicious for non-starters
                return player.role_classification == PlayerRole.STARTER
            
            total_snaps = player_snaps['snaps'].sum() if 'snaps' in player_snaps.columns else 0
            return total_snaps > 0
        
        return True
    
    def _has_significant_stats(self, stats: pd.Series) -> bool:
        """Check if player has meaningful statistical production"""
        
        # Passing stats
        if stats.get('passing_yards', 0) > 0 or stats.get('passing_attempts', 0) > 0:
            return True
            
        # Rushing stats  
        if stats.get('rushing_yards', 0) > 0 or stats.get('carries', 0) > 0:
            return True
            
        # Receiving stats
        if stats.get('receiving_yards', 0) > 0 or stats.get('targets', 0) > 0:
            return True
            
        return False
    
    def _create_validated_stat_record(self, stats: pd.Series, player: MasterPlayer) -> Dict:
        """Create validated stat record"""
        return {
            'player_id': player.nfl_id,
            'week': stats.get('week'),
            'season': stats.get('season'),
            'team': stats.get('recent_team'),
            'opponent': stats.get('opponent_team'),
            'position': player.position,
            'role_classification': player.role_classification.value,
            'snap_rate': player.avg_snap_rate_3_games,
            
            # Passing
            'passing_yards': stats.get('passing_yards', 0),
            'passing_attempts': stats.get('attempts', 0),
            'passing_completions': stats.get('completions', 0),
            'passing_tds': stats.get('passing_tds', 0),
            'interceptions': stats.get('interceptions', 0),
            
            # Rushing  
            'rushing_yards': stats.get('rushing_yards', 0),
            'carries': stats.get('carries', 0),
            'rushing_tds': stats.get('rushing_tds', 0),
            
            # Receiving
            'receiving_yards': stats.get('receiving_yards', 0),
            'targets': stats.get('targets', 0),
            'receptions': stats.get('receptions', 0),
            'receiving_tds': stats.get('receiving_tds', 0),
            
            # Fantasy
            'fantasy_points_ppr': stats.get('fantasy_points_ppr', 0),
            
            # Metadata
            'data_quality_score': player.data_quality_score,
            'last_validated': datetime.now(),
            'stats_validated': True,
            'role_at_time': player.role_classification.value
        }
