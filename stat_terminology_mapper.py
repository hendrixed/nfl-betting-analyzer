"""
Statistical Terminology Mapper

This module creates consistent mapping between different statistical terminology
used across NFL seasons and data sources to ensure uniform data processing.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import re

logger = logging.getLogger(__name__)

class StatTerminologyMapper:
    """Map statistical terminology across seasons and sources"""
    
    def __init__(self):
        # Define standardized terminology
        self.standard_terminology = {
            # Passing stats
            'passing_attempts': ['attempts', 'pass_att', 'passing_attempts', 'att', 'pass_attempts'],
            'passing_completions': ['completions', 'pass_comp', 'passing_completions', 'comp', 'pass_completions'],
            'passing_yards': ['passing_yards', 'pass_yds', 'pass_yards', 'passing_yds'],
            'passing_touchdowns': ['passing_tds', 'pass_td', 'passing_touchdowns', 'pass_tds', 'passing_td'],
            'passing_interceptions': ['interceptions', 'int', 'passing_int', 'ints', 'passing_interceptions'],
            'passing_first_downs': ['passing_first_downs', 'pass_fd', 'passing_fd', 'pass_first_downs'],
            
            # Rushing stats
            'rushing_attempts': ['carries', 'rush_att', 'rushing_attempts', 'carry', 'rushing_att'],
            'rushing_yards': ['rushing_yards', 'rush_yds', 'rush_yards', 'rushing_yds'],
            'rushing_touchdowns': ['rushing_tds', 'rush_td', 'rushing_touchdowns', 'rush_tds', 'rushing_td'],
            'rushing_first_downs': ['rushing_first_downs', 'rush_fd', 'rushing_fd', 'rush_first_downs'],
            'rushing_fumbles': ['rushing_fumbles', 'rush_fumbles', 'fumbles', 'rush_fum'],
            
            # Receiving stats
            'targets': ['targets', 'tgt', 'target', 'tgts'],
            'receptions': ['receptions', 'rec', 'catches', 'catch'],
            'receiving_yards': ['receiving_yards', 'rec_yds', 'receiving_yards', 'rec_yards'],
            'receiving_touchdowns': ['receiving_tds', 'rec_td', 'receiving_touchdowns', 'rec_tds', 'receiving_td'],
            'receiving_first_downs': ['receiving_first_downs', 'rec_fd', 'receiving_fd', 'rec_first_downs'],
            'receiving_fumbles': ['receiving_fumbles', 'rec_fumbles', 'rec_fum'],
            
            # Fantasy points
            'fantasy_points_ppr': ['fantasy_points_ppr', 'fantasy_points', 'ppr_points', 'fantasy_ppr'],
            'fantasy_points_standard': ['fantasy_points_standard', 'standard_points', 'fantasy_std'],
            'fantasy_points_half_ppr': ['fantasy_points_half_ppr', 'half_ppr_points', 'fantasy_half_ppr'],
            
            # Advanced stats
            'snap_count': ['snaps', 'snap_count', 'snap_number', 'snap_counts'],
            'snap_percentage': ['snap_pct', 'snap_percentage', 'snap_rate', 'snap_percent'],
            
            # Kicking stats
            'field_goals_made': ['fg_made', 'field_goals_made', 'fgm', 'fg_make'],
            'field_goals_attempted': ['fg_att', 'field_goals_attempted', 'fga', 'fg_attempt'],
            'extra_points_made': ['xp_made', 'extra_points_made', 'xpm', 'pat_made'],
            'extra_points_attempted': ['xp_att', 'extra_points_attempted', 'xpa', 'pat_att'],
            
            # Defense/Special Teams
            'sacks': ['sacks', 'sack', 'sk'],
            'tackles': ['tackles', 'tackle', 'tkl'],
            'interceptions_def': ['int_def', 'interceptions_def', 'def_int'],
            'fumbles_recovered': ['fumbles_recovered', 'fum_rec', 'fumble_rec'],
            'defensive_touchdowns': ['def_td', 'defensive_touchdowns', 'def_touchdowns']
        }
        
        # Position-specific required stats
        self.position_requirements = {
            'QB': ['passing_attempts', 'passing_completions', 'passing_yards', 'passing_touchdowns', 'passing_interceptions'],
            'RB': ['rushing_attempts', 'rushing_yards', 'rushing_touchdowns', 'targets', 'receptions', 'receiving_yards'],
            'WR': ['targets', 'receptions', 'receiving_yards', 'receiving_touchdowns'],
            'TE': ['targets', 'receptions', 'receiving_yards', 'receiving_touchdowns'],
            'K': ['field_goals_made', 'field_goals_attempted', 'extra_points_made'],
            'DEF': ['sacks', 'tackles', 'interceptions_def', 'fumbles_recovered']
        }
        
        # Season-specific variations (known differences across years)
        self.season_variations = {
            2020: {
                'fantasy_points': 'fantasy_points_ppr',
                'rec_yds': 'receiving_yards',
                'rush_yds': 'rushing_yards'
            },
            2021: {
                'fantasy_points': 'fantasy_points_ppr',
                'receiving_yds': 'receiving_yards',
                'rushing_yds': 'rushing_yards'
            },
            2022: {
                'fantasy_points_ppr': 'fantasy_points_ppr',
                'rec_yards': 'receiving_yards',
                'rush_yards': 'rushing_yards'
            },
            2023: {
                'fantasy_points_ppr': 'fantasy_points_ppr',
                'receiving_yards': 'receiving_yards',
                'rushing_yards': 'rushing_yards'
            },
            2024: {
                'fantasy_points_ppr': 'fantasy_points_ppr',
                'receiving_yards': 'receiving_yards',
                'rushing_yards': 'rushing_yards'
            }
        }
    
    def create_season_mapping(self, season: int, available_columns: List[str]) -> Dict[str, str]:
        """Create mapping from standardized terms to season-specific column names"""
        
        logger.info(f"Creating terminology mapping for {season} season")
        
        mapping = {}
        available_lower = [col.lower() for col in available_columns]
        
        # Apply season-specific variations first
        season_vars = self.season_variations.get(season, {})
        
        for standard_term, variations in self.standard_terminology.items():
            best_match = None
            
            # Check season-specific variations first
            if standard_term in season_vars.values():
                for orig_term, std_term in season_vars.items():
                    if std_term == standard_term and orig_term.lower() in available_lower:
                        idx = available_lower.index(orig_term.lower())
                        best_match = available_columns[idx]
                        break
            
            # If no season-specific match, use general variations
            if not best_match:
                for variation in variations:
                    variation_lower = variation.lower()
                    
                    # Exact match
                    if variation_lower in available_lower:
                        idx = available_lower.index(variation_lower)
                        best_match = available_columns[idx]
                        break
                    
                    # Partial match with fuzzy matching
                    if not best_match:
                        for col, col_lower in zip(available_columns, available_lower):
                            # Check if variation is contained in column name
                            if variation_lower in col_lower:
                                best_match = col
                                break
                            # Check if column name is contained in variation (for abbreviations)
                            elif col_lower in variation_lower and len(col_lower) >= 3:
                                best_match = col
                                break
            
            if best_match:
                mapping[standard_term] = best_match
                logger.debug(f"Mapped {standard_term} -> {best_match}")
            else:
                logger.warning(f"No mapping found for {standard_term} in {season} data")
        
        logger.info(f"Created mapping for {len(mapping)} statistical terms")
        return mapping
    
    def validate_position_coverage(self, position: str, available_stats: List[str]) -> Tuple[bool, List[str]]:
        """Validate that all required stats are available for a position"""
        
        required_stats = self.position_requirements.get(position, [])
        missing_stats = [stat for stat in required_stats if stat not in available_stats]
        
        is_complete = len(missing_stats) == 0
        return is_complete, missing_stats
    
    def get_fantasy_point_calculation_mapping(self) -> Dict[str, Dict[str, float]]:
        """Get standardized fantasy point calculation coefficients"""
        
        return {
            'standard': {
                'passing_yards': 0.04,      # 1 pt per 25 yards
                'passing_touchdowns': 4.0,
                'passing_interceptions': -2.0,
                'rushing_yards': 0.1,       # 1 pt per 10 yards
                'rushing_touchdowns': 6.0,
                'receiving_yards': 0.1,     # 1 pt per 10 yards
                'receiving_touchdowns': 6.0,
                'receptions': 0.0,          # No points for receptions
                'fumbles_lost': -2.0
            },
            'ppr': {
                'passing_yards': 0.04,
                'passing_touchdowns': 4.0,
                'passing_interceptions': -2.0,
                'rushing_yards': 0.1,
                'rushing_touchdowns': 6.0,
                'receiving_yards': 0.1,
                'receiving_touchdowns': 6.0,
                'receptions': 1.0,          # 1 point per reception
                'fumbles_lost': -2.0
            },
            'half_ppr': {
                'passing_yards': 0.04,
                'passing_touchdowns': 4.0,
                'passing_interceptions': -2.0,
                'rushing_yards': 0.1,
                'rushing_touchdowns': 6.0,
                'receiving_yards': 0.1,
                'receiving_touchdowns': 6.0,
                'receptions': 0.5,          # 0.5 points per reception
                'fumbles_lost': -2.0
            }
        }
    
    def normalize_column_name(self, column_name: str) -> str:
        """Normalize column name for consistent matching"""
        
        if not column_name:
            return ""
        
        # Convert to lowercase
        normalized = column_name.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['player_', 'team_', 'game_']
        suffixes_to_remove = ['_total', '_game', '_season']
        
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Replace common separators with underscores
        normalized = re.sub(r'[-\s]+', '_', normalized)
        
        # Remove multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        
        return normalized.strip('_')
    
    def get_stat_category(self, stat_name: str) -> str:
        """Determine the category of a statistical field"""
        
        stat_lower = stat_name.lower()
        
        if any(term in stat_lower for term in ['pass', 'completion', 'attempt', 'int']):
            return 'passing'
        elif any(term in stat_lower for term in ['rush', 'carry', 'carries']):
            return 'rushing'
        elif any(term in stat_lower for term in ['rec', 'target', 'catch']):
            return 'receiving'
        elif any(term in stat_lower for term in ['fantasy', 'ppr']):
            return 'fantasy'
        elif any(term in stat_lower for term in ['snap', 'play']):
            return 'usage'
        elif any(term in stat_lower for term in ['fg', 'field_goal', 'extra_point', 'xp']):
            return 'kicking'
        elif any(term in stat_lower for term in ['sack', 'tackle', 'def', 'fumble']):
            return 'defense'
        else:
            return 'other'
    
    def validate_mapping_completeness(self, mapping: Dict[str, str], position: str) -> Dict[str, Any]:
        """Validate completeness of mapping for a specific position"""
        
        required_stats = self.position_requirements.get(position, [])
        mapped_stats = list(mapping.keys())
        
        missing_required = [stat for stat in required_stats if stat not in mapped_stats]
        extra_mapped = [stat for stat in mapped_stats if stat not in required_stats]
        
        completeness_score = len([s for s in required_stats if s in mapped_stats]) / len(required_stats) if required_stats else 1.0
        
        return {
            'completeness_score': completeness_score,
            'missing_required_stats': missing_required,
            'extra_mapped_stats': extra_mapped,
            'total_mapped': len(mapped_stats),
            'is_complete': len(missing_required) == 0
        }
    
    def create_reverse_mapping(self, forward_mapping: Dict[str, str]) -> Dict[str, str]:
        """Create reverse mapping from original column names to standardized terms"""
        
        return {v: k for k, v in forward_mapping.items()}
    
    def get_all_possible_variations(self, standard_term: str) -> List[str]:
        """Get all possible variations for a standardized term"""
        
        return self.standard_terminology.get(standard_term, [standard_term])
