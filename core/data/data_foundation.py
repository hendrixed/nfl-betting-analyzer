"""
NFL Data Foundation - Core Data Structures for Enhanced Architecture

This module defines the fundamental data structures for the new hierarchical
NFL data collection system that ensures accurate starter identification and
proper validation of player statistics.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, date


class PlayerRole(Enum):
    """Player role classification based on depth chart and usage"""
    STARTER = "starter"
    BACKUP_HIGH = "backup_high"      # Key backup, likely to play
    BACKUP_LOW = "backup_low"        # Deep bench
    INACTIVE = "inactive"            # Not active this week
    SPECIAL_TEAMS = "special_teams"  # ST only


@dataclass
class MasterPlayer:
    """Authoritative player record combining all data sources"""
    # CRITICAL: Use official NFL identifiers as primary keys
    nfl_id: str                      # Primary - official NFL player ID
    gsis_id: Optional[str] = None    # Game Statistics ID
    pfr_id: Optional[str] = None     # Pro Football Reference ID
    espn_id: Optional[str] = None    # ESPN player ID
    
    # Basic Info
    name: str = ""
    first_name: str = ""
    last_name: str = ""
    position: str = ""
    
    # Compatibility fields (for tests and legacy code)
    team: Optional[str] = None                 # alias of current_team
    jersey_number: Optional[int] = None
    height: Optional[int] = None               # inches
    weight: Optional[int] = None               # lbs
    college: Optional[str] = None
    years_pro: Optional[int] = None
    birth_date: Optional[datetime] = None
    is_active: Optional[bool] = None
    
    # Current Status (changes weekly)
    current_team: str = ""
    roster_status: str = ""               # ACT/INA/RES/IR/PUP/etc
    depth_chart_rank: Optional[int] = None  # 1=starter, 2=first backup, etc
    role_classification: Optional[PlayerRole] = None
    
    # Recent Performance Metrics
    avg_snap_rate_3_games: float = 0.0      # Recent snap rate
    games_played_season: int = 0
    is_injured: bool = False
    
    # Data Quality
    data_quality_score: float = 0.0         # 0-1 based on data completeness
    last_validated: Optional[datetime] = None
    inconsistency_flags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Maintain team/current_team compatibility
        if self.team and not self.current_team:
            self.current_team = self.team
        if self.current_team and not self.team:
            self.team = self.current_team


@dataclass
class WeeklyRosterSnapshot:
    """Complete roster state for a specific week"""
    season: int
    week: int
    team: str
    snapshot_date: datetime
    
    # Players by Role (CRITICAL: Only include players likely to have meaningful stats)
    starters: List[MasterPlayer] = field(default_factory=list)           # Expected to play 70%+ snaps
    backup_primary: List[MasterPlayer] = field(default_factory=list)     # Key backups, 30-70% snaps
    backup_depth: List[MasterPlayer] = field(default_factory=list)       # Emergency players, <30% snaps
    inactive: List[MasterPlayer] = field(default_factory=list)           # Will not play this week
    
    # Validation Metrics
    depth_chart_confidence: float = 0.0    # How reliable is this depth chart
    injury_impact_score: float = 0.0       # How much injuries changed depth
    
    def get_active_players(self) -> List[MasterPlayer]:
        """Get all players expected to play (starters + backups)"""
        return self.starters + self.backup_primary + self.backup_depth
    
    def get_stat_eligible_players(self) -> List[MasterPlayer]:
        """Get players who should have stats tracked"""
        return self.starters + self.backup_primary


@dataclass
class ValidationReport:
    """Report on data quality and consistency"""
    depth_chart_accuracy: float = 0.0
    stats_snap_consistency: float = 0.0
    player_id_consistency: float = 0.0
    roster_completeness: float = 0.0
    overall_score: float = 0.0
    issues_found: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.overall_score = (
            self.depth_chart_accuracy + 
            self.stats_snap_consistency + 
            self.player_id_consistency + 
            self.roster_completeness
        ) / 4


@dataclass
class PlayerGameValidation:
    """Validation data for a player's game performance"""
    player_id: str
    week: int
    team: str
    
    # Snap validation
    expected_snaps: int = 0
    actual_snaps: int = 0
    snap_rate: float = 0.0
    
    # Role validation
    expected_role: Optional[PlayerRole] = None
    actual_usage: str = ""  # Based on stats
    
    # Consistency flags
    has_stats_without_snaps: bool = False
    has_snaps_without_stats: bool = False
    role_mismatch: bool = False
    
    # Quality score
    validation_score: float = 0.0  # 0-1 based on consistency


@dataclass
class TeamDepthChart:
    """Structured depth chart for a team at a specific position"""
    team: str
    position: str
    week: int
    
    # Ordered by depth (1st string, 2nd string, etc.)
    depth_order: List[MasterPlayer] = field(default_factory=list)
    
    # Confidence metrics
    source_reliability: float = 0.0  # How reliable is the source
    injury_adjustments: List[str] = field(default_factory=list)  # Injury-based changes
    last_updated: Optional[datetime] = None
    
    def get_starter(self) -> Optional[MasterPlayer]:
        """Get the expected starter at this position"""
        return self.depth_order[0] if self.depth_order else None
    
    def get_primary_backup(self) -> Optional[MasterPlayer]:
        """Get the primary backup at this position"""
        return self.depth_order[1] if len(self.depth_order) > 1 else None


@dataclass
class DataSourceMetrics:
    """Track reliability and consistency of different data sources"""
    source_name: str
    
    # Reliability metrics
    uptime_percentage: float = 0.0
    data_freshness_hours: float = 0.0
    consistency_score: float = 0.0
    
    # Coverage metrics
    players_covered: int = 0
    teams_covered: int = 0
    positions_covered: List[str] = field(default_factory=list)
    
    # Quality indicators
    missing_data_rate: float = 0.0
    conflicting_data_rate: float = 0.0
    
    last_assessment: Optional[datetime] = None


@dataclass
class WeeklyDataQualityReport:
    """Comprehensive data quality report for a week"""
    season: int
    week: int
    generated_at: datetime
    
    # Overall metrics
    total_players_processed: int = 0
    players_with_high_quality: int = 0
    teams_with_complete_rosters: int = 0
    
    # Source reliability
    source_metrics: Dict[str, DataSourceMetrics] = field(default_factory=dict)
    
    # Validation results
    validation_report: Optional[ValidationReport] = None
    
    # Critical issues
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    def get_overall_quality_score(self) -> float:
        """Calculate overall data quality score"""
        if self.total_players_processed == 0:
            return 0.0
        
        quality_ratio = self.players_with_high_quality / self.total_players_processed
        validation_score = self.validation_report.overall_score if self.validation_report else 0.0
        
        return (quality_ratio + validation_score) / 2


# Utility functions for data structure manipulation
def merge_player_data(primary: MasterPlayer, secondary: MasterPlayer) -> MasterPlayer:
    """Merge two player records, prioritizing primary source"""
    merged = MasterPlayer(nfl_id=primary.nfl_id)
    
    # Use primary data where available, fall back to secondary
    for field_name in primary.__dataclass_fields__:
        primary_value = getattr(primary, field_name)
        secondary_value = getattr(secondary, field_name)
        
        if primary_value and primary_value != "":
            setattr(merged, field_name, primary_value)
        elif secondary_value and secondary_value != "":
            setattr(merged, field_name, secondary_value)
    
    # Combine inconsistency flags
    merged.inconsistency_flags = list(set(
        primary.inconsistency_flags + secondary.inconsistency_flags
    ))
    
    return merged


def classify_position_group(position: str) -> str:
    """Classify position into broader groups for analysis"""
    position = position.upper()
    
    if position == 'QB':
        return 'QUARTERBACK'
    elif position in ['RB', 'FB']:
        return 'RUNNING_BACK'
    elif position in ['WR', 'WR1', 'WR2', 'WR3', 'SWR']:
        return 'WIDE_RECEIVER'
    elif position in ['TE', 'TE1', 'TE2']:
        return 'TIGHT_END'
    elif position in ['K', 'PK']:
        return 'KICKER'
    elif position in ['P']:
        return 'PUNTER'
    elif position in ['DST', 'DEF']:
        return 'DEFENSE'
    else:
        return 'OTHER'


def validate_player_role_consistency(player: MasterPlayer) -> List[str]:
    """Validate that a player's role classification is consistent with other data"""
    issues = []
    
    if not player.role_classification:
        issues.append("Missing role classification")
        return issues
    
    # Check depth chart consistency
    if player.depth_chart_rank and player.role_classification:
        if player.depth_chart_rank == 1 and player.role_classification != PlayerRole.STARTER:
            if player.avg_snap_rate_3_games < 0.3:  # Unless they're not playing
                issues.append(f"Depth chart rank 1 but classified as {player.role_classification.value}")
        
        elif player.depth_chart_rank > 2 and player.role_classification == PlayerRole.STARTER:
            issues.append(f"Classified as starter but depth chart rank {player.depth_chart_rank}")
    
    # Check snap rate consistency
    if player.role_classification == PlayerRole.STARTER and player.avg_snap_rate_3_games < 0.4:
        issues.append(f"Starter with low snap rate: {player.avg_snap_rate_3_games:.2f}")
    
    elif player.role_classification == PlayerRole.INACTIVE and player.avg_snap_rate_3_games > 0.1:
        issues.append(f"Inactive player with significant snaps: {player.avg_snap_rate_3_games:.2f}")
    
    # Check injury status consistency
    if player.is_injured and player.role_classification in [PlayerRole.STARTER, PlayerRole.BACKUP_HIGH]:
        if player.avg_snap_rate_3_games > 0.2:
            issues.append("Injured player with high expected usage")
    
    return issues
