"""
Data Quality Validator - Comprehensive validation and quality checking

This module provides comprehensive data validation and quality checking
for the enhanced NFL data architecture to ensure data consistency
and reliability across all sources.
"""

import logging
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime
from core.data.data_foundation import (
    WeeklyRosterSnapshot, ValidationReport, PlayerRole, MasterPlayer,
    PlayerGameValidation, WeeklyDataQualityReport
)

logger = logging.getLogger(__name__)


# Simple validation result types expected by tests
class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    is_valid: bool
    severity: ValidationSeverity
    score: float
    message: str = ""


class DataQualityValidator:
    """Validate data consistency and quality across sources"""
    
    def __init__(self, session=None):
        self.session = session
        # Expose rules per tests' expectations
        self.validation_rules: Dict[str, Any] = {
            'min_starters_per_position': {'QB': 1, 'RB': 1, 'WR': 2, 'TE': 1},
            'max_starters_per_position': {'QB': 2, 'RB': 3, 'WR': 4, 'TE': 2},
            'min_snap_rate_starter': 0.5,
            'max_depth_rank_starter': 2,
            'min_data_quality_score': 0.6
        }
        # Keep original thresholds name for internal use
        self.validation_thresholds: Dict[str, Any] = self.validation_rules

    def validate_player_completeness(self, player: MasterPlayer) -> ValidationResult:
        """Basic completeness check for a MasterPlayer. Returns ValidationResult.
        Fields considered: jersey_number, height, weight, college, years_pro, birth_date.
        """
        total = 6
        present = 0
        present += 1 if getattr(player, 'jersey_number', None) not in (None, '') else 0
        present += 1 if getattr(player, 'height', None) not in (None, '') else 0
        present += 1 if getattr(player, 'weight', None) not in (None, '') else 0
        present += 1 if getattr(player, 'college', None) not in (None, '') else 0
        present += 1 if getattr(player, 'years_pro', None) not in (None, '') else 0
        present += 1 if getattr(player, 'birth_date', None) not in (None, '') else 0

        score = present / total if total else 1.0
        if score >= 0.9:
            return ValidationResult(True, ValidationSeverity.INFO, score, "Player record complete")
        elif score >= 0.75:
            return ValidationResult(False, ValidationSeverity.WARNING, score, "Player record has missing optional fields")
        else:
            return ValidationResult(False, ValidationSeverity.ERROR, score, "Player record incomplete")
    
    def validate_weekly_snapshots(self, snapshots: Dict[str, WeeklyRosterSnapshot]) -> ValidationReport:
        """Run comprehensive validation on weekly roster snapshots"""
        
        logger.info(f"Validating {len(snapshots)} team roster snapshots")
        
        report = ValidationReport()
        all_issues = []
        
        # Individual team validations
        team_scores = []
        for team, snapshot in snapshots.items():
            team_validation = self._validate_team_snapshot(team, snapshot)
            team_scores.append(team_validation)
            
            if team_validation < 0.7:
                all_issues.append(f"Team {team} has low validation score: {team_validation:.2f}")
        
        # Calculate overall scores
        report.roster_completeness = float(np.mean(team_scores)) if team_scores else 0.0
        report.depth_chart_accuracy = self._validate_depth_chart_consistency(snapshots)
        report.stats_snap_consistency = self._validate_stats_snap_alignment(snapshots)
        report.player_id_consistency = self._validate_player_id_consistency(snapshots)
        
        report.issues_found = all_issues
        
        logger.info(f"Validation complete. Overall score: {report.overall_score:.2f}")
        return report
    
    def _validate_team_snapshot(self, team: str, snapshot: WeeklyRosterSnapshot) -> float:
        """Validate a single team's roster snapshot"""
        
        score = 1.0
        issues = []
        
        # Check position coverage
        all_players = snapshot.get_active_players()
        position_counts = {}
        
        for player in all_players:
            pos = player.position
            if pos not in position_counts:
                position_counts[pos] = {'starters': 0, 'total': 0}
            
            position_counts[pos]['total'] += 1
            if player.role_classification == PlayerRole.STARTER:
                position_counts[pos]['starters'] += 1
        
        # Validate position requirements
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos not in position_counts:
                score -= 0.2
                issues.append(f"{team} missing {pos} players")
                continue
            
            starters = position_counts[pos]['starters']
            min_starters = self.validation_thresholds['min_starters_per_position'][pos]
            max_starters = self.validation_thresholds['max_starters_per_position'][pos]
            
            if starters < min_starters:
                score -= 0.1
                issues.append(f"{team} has only {starters} {pos} starters (min: {min_starters})")
            elif starters > max_starters:
                score -= 0.05
                issues.append(f"{team} has {starters} {pos} starters (max: {max_starters})")
        
        # Check data quality scores
        low_quality_players = [p for p in all_players 
                             if p.data_quality_score < self.validation_thresholds['min_data_quality_score']]
        
        if low_quality_players:
            quality_penalty = len(low_quality_players) * 0.02
            score -= quality_penalty
            issues.append(f"{team} has {len(low_quality_players)} low-quality player records")
        
        if issues:
            logger.warning(f"Team {team} validation issues: {issues}")
        
        return max(score, 0.0)
    
    def _validate_depth_chart_consistency(self, snapshots: Dict[str, WeeklyRosterSnapshot]) -> float:
        """Check if depth charts are consistent with usage patterns"""
        
        consistency_scores = []
        
        for team, snapshot in snapshots.items():
            team_score = 0.0
            position_scores = []
            
            for position in ['QB', 'RB', 'WR', 'TE']:
                pos_players = [p for p in snapshot.get_active_players() if p.position == position]
                
                if not pos_players:
                    continue
                
                # Sort by depth chart rank, then by snap rate
                pos_players.sort(key=lambda x: (x.depth_chart_rank or 99, -x.avg_snap_rate_3_games))
                
                # Check if higher depth chart ranks have higher snap rates
                rank_correlation = self._calculate_rank_correlation(
                    [p.depth_chart_rank or 99 for p in pos_players],
                    [p.avg_snap_rate_3_games for p in pos_players]
                )
                
                position_scores.append(max(rank_correlation, 0.0))
            
            if position_scores:
                team_score = float(np.mean(position_scores))
                consistency_scores.append(team_score)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    def _calculate_rank_correlation(self, ranks: List[int], values: List[float]) -> float:
        """Calculate correlation between ranks and values (negative correlation expected)"""
        if len(ranks) < 2 or len(values) < 2:
            return 1.0
        
        try:
            correlation = np.corrcoef(ranks, values)[0, 1]
            # We want negative correlation (lower rank = higher value)
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _validate_stats_snap_alignment(self, snapshots: Dict[str, WeeklyRosterSnapshot]) -> float:
        """Validate that players with stats actually had playing time"""
        # This would require access to actual stats data
        # For now, return based on role classifications being reasonable
        
        alignment_scores: List[float] = []
        
        for team, snapshot in snapshots.items():
            # Check if role classifications are reasonable
            starters = len(snapshot.starters)
            backups = len(snapshot.backup_primary)
            
            # Reasonable ratios
            if 4 <= starters <= 12 and backups <= starters * 2:
                alignment_scores.append(1.0)
            else:
                alignment_scores.append(0.5)
        
        return float(np.mean(alignment_scores)) if alignment_scores else 0.0
    
    def _validate_player_id_consistency(self, snapshots: Dict[str, WeeklyRosterSnapshot]) -> float:
        """Check for player ID consistency across teams"""
        
        all_player_ids: Set[str] = set()
        duplicate_count = 0
        
        for team, snapshot in snapshots.items():
            team_player_ids = {p.nfl_id for p in snapshot.get_active_players()}
            
            # Check for duplicates across teams (same player on multiple teams)
            duplicates = all_player_ids.intersection(team_player_ids)
            duplicate_count += len(duplicates)
            
            all_player_ids.update(team_player_ids)
        
        if not all_player_ids:
            return 0.0
        
        consistency_score = 1.0 - (duplicate_count / len(all_player_ids))
        return max(consistency_score, 0.0)


class StatsValidator:
    """Validate statistical data against roster and snap data"""
    
    def __init__(self, session=None):
        self.session = session
        self.validation_rules = {
            'max_passing_yards_per_attempt': 25.0,
            'max_rushing_yards_per_carry': 15.0,
            'max_receiving_yards_per_target': 20.0,
            'min_snap_rate_for_stats': 0.1
        }
        # Position ranges (for extensibility in other tests)
        self.position_stat_ranges = {
            'QB': {'pass_yds_per_att': (0, 25)},
            'RB': {'rush_yds_per_carry': (0, 15)},
            'WR': {'rec_yds_per_target': (0, 20)},
            'TE': {'rec_yds_per_target': (0, 20)},
        }
    
    def validate_player_stats(self, stats: Dict, player: MasterPlayer, 
                            snap_data: Optional[Dict] = None) -> PlayerGameValidation:
        """Validate individual player statistics"""
        
        validation = PlayerGameValidation(
            player_id=player.nfl_id,
            week=stats.get('week', 0),
            team=stats.get('team', ''),
            expected_role=player.role_classification
        )
        
        # Check snap consistency
        if snap_data:
            validation.actual_snaps = snap_data.get('snaps', 0)
            validation.snap_rate = snap_data.get('snap_pct', 0.0)
            
            # Flag inconsistencies
            if self._has_significant_stats(stats) and validation.actual_snaps == 0:
                validation.has_stats_without_snaps = True
            
            if validation.actual_snaps > 10 and not self._has_significant_stats(stats):
                validation.has_snaps_without_stats = True
        
        # Check role consistency
        actual_usage = self._classify_usage_from_stats(stats)
        validation.actual_usage = actual_usage
        
        if player.role_classification == PlayerRole.INACTIVE and self._has_significant_stats(stats):
            validation.role_mismatch = True
        
        # Calculate validation score
        validation.validation_score = self._calculate_validation_score(validation, stats)
        
        return validation

    def validate_fantasy_points_calculation(self, stat) -> ValidationResult:
        """Validate PPR fantasy points calculation for a PlayerGameStats-like object.
        Expected formula (simplified PPR):
          pass: yards*0.04 + TDs*4 - INT*2
          rush: yards*0.1 + TDs*6
          rec:  yards*0.1 + TDs*6 + receptions*1
        """
        try:
            def _to_float_safe(val) -> float:
                try:
                    if isinstance(val, (int, float, np.number)):
                        return float(val)
                    # Treat mocks/None/empty as 0
                    return 0.0
                except Exception:
                    return 0.0

            pass_yards = _to_float_safe(getattr(stat, 'passing_yards', 0))
            pass_tds = _to_float_safe(getattr(stat, 'passing_touchdowns', 0))
            # Avoid default-evaluated getattr producing a Mock by reading separately
            pi_attr = getattr(stat, 'passing_interceptions', None)
            if pi_attr is None:
                pi_attr = getattr(stat, 'interceptions', 0)
            ints = _to_float_safe(pi_attr)
            rush_yards = _to_float_safe(getattr(stat, 'rushing_yards', 0))
            rush_tds = _to_float_safe(getattr(stat, 'rushing_touchdowns', 0))
            rec_yards = _to_float_safe(getattr(stat, 'receiving_yards', 0))
            rec_tds = _to_float_safe(getattr(stat, 'receiving_touchdowns', 0))
            receptions = _to_float_safe(getattr(stat, 'receptions', 0))

            expected = (
                pass_yards * 0.04 + pass_tds * 4 - ints * 2
                + rush_yards * 0.1 + rush_tds * 6
                + rec_yards * 0.1 + rec_tds * 6 + receptions * 1.0
            )
            actual = _to_float_safe(getattr(stat, 'fantasy_points_ppr', 0))
            delta = abs(expected - actual)
            if delta <= 0.01:
                return ValidationResult(True, ValidationSeverity.INFO, 1.0, "Fantasy points calculation correct")
            else:
                return ValidationResult(False, ValidationSeverity.WARNING, max(0.0, 1.0 - delta/10.0), f"Expected {expected:.2f}, got {actual:.2f}")
        except Exception as e:
            return ValidationResult(False, ValidationSeverity.ERROR, 0.0, f"Validation failed: {e}")
    
    def _has_significant_stats(self, stats: Dict) -> bool:
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
    
    def _classify_usage_from_stats(self, stats: Dict) -> str:
        """Classify player usage based on statistical production"""
        
        total_touches = (stats.get('carries', 0) + 
                        stats.get('targets', 0) + 
                        stats.get('passing_attempts', 0))
        
        if total_touches >= 15:
            return "heavy_usage"
        elif total_touches >= 8:
            return "moderate_usage"
        elif total_touches >= 3:
            return "light_usage"
        else:
            return "minimal_usage"
    
    def _calculate_validation_score(self, validation: PlayerGameValidation, stats: Dict) -> float:
        """Calculate overall validation score for player"""
        
        score = 1.0
        
        # Penalize inconsistencies
        if validation.has_stats_without_snaps:
            score -= 0.3
        
        if validation.has_snaps_without_stats:
            score -= 0.2
        
        if validation.role_mismatch:
            score -= 0.4
        
        # Check statistical reasonableness
        if not self._are_stats_reasonable(stats):
            score -= 0.2
        
        return max(score, 0.0)
    
    def _are_stats_reasonable(self, stats: Dict) -> bool:
        """Check if statistics are within reasonable bounds"""
        
        # Check passing efficiency
        attempts = stats.get('passing_attempts', 0)
        if attempts > 0:
            yards_per_attempt = stats.get('passing_yards', 0) / attempts
            if yards_per_attempt > self.validation_rules['max_passing_yards_per_attempt']:
                return False
        
        # Check rushing efficiency
        carries = stats.get('carries', 0)
        if carries > 0:
            yards_per_carry = stats.get('rushing_yards', 0) / carries
            if yards_per_carry > self.validation_rules['max_rushing_yards_per_carry']:
                return False
        
        # Check receiving efficiency
        targets = stats.get('targets', 0)
        if targets > 0:
            yards_per_target = stats.get('receiving_yards', 0) / targets
            if yards_per_target > self.validation_rules['max_receiving_yards_per_target']:
                return False
        
        return True


class ComprehensiveValidator:
    """Main validator that orchestrates all validation processes"""
    
    def __init__(self):
        self.quality_validator = DataQualityValidator()
        self.stats_validator = StatsValidator()
    
    def run_full_validation(self, season: int, week: int,
                          snapshots: Dict[str, WeeklyRosterSnapshot],
                          stats_data: List[Dict],
                          snap_data: List[Dict]) -> WeeklyDataQualityReport:
        """Run comprehensive validation for a week"""
        
        logger.info(f"Running full validation for {season} Week {week}")
        
        # Create quality report
        report = WeeklyDataQualityReport(
            season=season,
            week=week,
            generated_at=datetime.now()
        )
        
        # Validate roster snapshots
        validation_report = self.quality_validator.validate_weekly_snapshots(snapshots)
        report.validation_report = validation_report
        
        # Count processed players
        total_players = sum(len(snapshot.get_active_players()) for snapshot in snapshots.values())
        high_quality_players = sum(
            len([p for p in snapshot.get_active_players() if p.data_quality_score >= 0.8])
            for snapshot in snapshots.values()
        )
        
        report.total_players_processed = total_players
        report.players_with_high_quality = high_quality_players
        report.teams_with_complete_rosters = len([s for s in snapshots.values() if len(s.starters) >= 4])
        
        # Validate individual player stats
        player_validations = []
        for stats in stats_data:
            player_id = stats.get('player_id')
            if not player_id:
                continue
            
            # Find player in snapshots
            player = self._find_player_in_snapshots(player_id, snapshots)
            if not player:
                continue
            
            # Find snap data
            snap_info = next((s for s in snap_data if s.get('player_id') == player_id), None)
            
            # Validate
            validation = self.stats_validator.validate_player_stats(stats, player, snap_info)
            player_validations.append(validation)
        
        # Analyze validation results
        if player_validations:
            avg_validation_score = float(np.mean([v.validation_score for v in player_validations]))
            report.validation_report.stats_snap_consistency = avg_validation_score
        
        # Generate recommendations
        report.recommended_actions = self._generate_recommendations(report)
        
        logger.info(f"Validation complete. Overall quality score: {report.get_overall_quality_score():.2f}")
        return report
    
    def _find_player_in_snapshots(self, player_id: str, 
                                snapshots: Dict[str, WeeklyRosterSnapshot]) -> Optional[MasterPlayer]:
        """Find a player in the roster snapshots"""
        
        for snapshot in snapshots.values():
            for player in snapshot.get_active_players():
                if player.nfl_id == player_id:
                    return player
        return None
    
    def _generate_recommendations(self, report: WeeklyDataQualityReport) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        
        recommendations = []
        
        if report.get_overall_quality_score() < 0.7:
            recommendations.append("Overall data quality is below acceptable threshold - review data sources")
        
        if report.validation_report and report.validation_report.roster_completeness < 0.8:
            recommendations.append("Roster data is incomplete - verify roster source reliability")
        
        if report.validation_report and report.validation_report.depth_chart_accuracy < 0.7:
            recommendations.append("Depth chart data inconsistent with usage - update depth chart sources")
        
        if report.validation_report and report.validation_report.stats_snap_consistency < 0.8:
            recommendations.append("Stats and snap data misaligned - implement stricter validation rules")
        
        if report.total_players_processed > 0:
            quality_ratio = report.players_with_high_quality / report.total_players_processed
            if quality_ratio < 0.8:
                recommendations.append("High number of low-quality player records - improve data collection")
        
        return recommendations
