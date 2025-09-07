"""
Player Identity Resolver

This module resolves player identity conflicts across multiple seasons
and creates consistent player ID mapping to ensure accurate historical tracking.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import difflib
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class PlayerIdentityConflict:
    """Represents a player identity conflict requiring resolution"""
    conflicting_ids: List[str]
    player_names: List[str]
    positions: List[str]
    teams: List[str]
    seasons: List[int]
    confidence_score: float  # 0-1, how confident we are these are the same player
    resolution_method: str   # How the conflict was resolved

@dataclass 
class PlayerMasterRecord:
    """Master record for a player across all seasons"""
    master_id: str           # Canonical player ID
    canonical_name: str      # Most common/recent name
    primary_position: str    # Most common position
    all_ids: List[str]       # All IDs used for this player
    all_names: List[str]     # All name variations
    seasons_active: List[int] # Seasons where player appeared
    teams_played_for: List[str] # All teams
    confidence_score: float  # Overall confidence in identity resolution
    birth_date: Optional[str] = None  # If available
    college: Optional[str] = None     # If available

class PlayerIdentityResolver:
    """Resolve player identity conflicts and create consistent mapping"""
    
    def __init__(self):
        self.similarity_threshold = 0.85  # Name similarity threshold
        self.position_weight = 0.3       # Weight for position matching
        self.team_weight = 0.2          # Weight for team history
        self.temporal_weight = 0.2      # Weight for temporal overlap
        self.name_weight = 0.3          # Weight for name similarity
        
        # Common name variations and nicknames
        self.name_variations = {
            'jr': ['jr', 'jr.', 'junior'],
            'sr': ['sr', 'sr.', 'senior'],
            'iii': ['iii', '3rd', 'third'],
            'ii': ['ii', '2nd', 'second'],
            'iv': ['iv', '4th', 'fourth'],
        }
        
        # Position compatibility groups
        self.position_groups = {
            'skill_positions': ['QB', 'RB', 'WR', 'TE'],
            'offensive_line': ['C', 'G', 'T', 'OL'],
            'defensive_line': ['DE', 'DT', 'NT', 'DL'],
            'linebackers': ['LB', 'ILB', 'OLB', 'MLB'],
            'defensive_backs': ['CB', 'S', 'FS', 'SS', 'DB'],
            'special_teams': ['K', 'P', 'LS']
        }
        
    def create_identity_mapping(self, player_data: pd.DataFrame) -> Dict[str, str]:
        """
        Create mapping from all player IDs to canonical master IDs
        Returns: Dict[original_id, master_id]
        """
        
        logger.info(f"Resolving player identities for {len(player_data)} player records")
        
        # Step 1: Clean and standardize player data
        cleaned_data = self._clean_player_data(player_data)
        
        # Step 2: Identify potential conflicts
        conflicts = self._identify_identity_conflicts(cleaned_data)
        logger.info(f"Found {len(conflicts)} potential identity conflicts")
        
        # Step 3: Resolve conflicts using multiple strategies
        resolved_players = self._resolve_identity_conflicts(conflicts, cleaned_data)
        logger.info(f"Resolved into {len(resolved_players)} master player records")
        
        # Step 4: Create mapping
        identity_mapping = self._create_id_mapping(resolved_players)
        logger.info(f"Created identity mapping for {len(identity_mapping)} player IDs")
        
        # Step 5: Validate mapping quality
        validation_report = self._validate_mapping_quality(identity_mapping, cleaned_data)
        logger.info(f"Mapping validation: {validation_report['quality_score']:.3f} quality score")
        
        return identity_mapping
    
    def _clean_player_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize player data for consistent processing"""
        
        cleaned = player_data.copy()
        
        # Standardize column names
        column_mapping = {
            'name': 'player_name',
            'full_name': 'player_name',
            'display_name': 'player_name',
            'team': 'recent_team',
            'current_team': 'recent_team'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in cleaned.columns and new_col not in cleaned.columns:
                cleaned[new_col] = cleaned[old_col]
        
        # Clean player names
        if 'player_name' in cleaned.columns:
            cleaned['player_name'] = cleaned['player_name'].apply(self._clean_name)
            cleaned['name_normalized'] = cleaned['player_name'].apply(self._normalize_name)
        
        # Standardize positions
        if 'position' in cleaned.columns:
            cleaned['position'] = cleaned['position'].apply(self._standardize_position)
        
        # Ensure required columns exist
        required_columns = ['player_id', 'player_name', 'position', 'recent_team', 'season']
        for col in required_columns:
            if col not in cleaned.columns:
                cleaned[col] = None
        
        # Remove records with missing critical data
        cleaned = cleaned.dropna(subset=['player_id', 'player_name'])
        
        logger.info(f"Cleaned data: {len(cleaned)} records with valid player_id and name")
        return cleaned
    
    def _clean_name(self, name: str) -> str:
        """Clean player name for comparison"""
        if not name or pd.isna(name):
            return ""
        
        # Convert to string and strip whitespace
        cleaned = str(name).strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Handle common formatting issues
        cleaned = cleaned.replace('.', '')  # Remove periods
        cleaned = re.sub(r'[^\w\s\-\']', '', cleaned)  # Keep only alphanumeric, spaces, hyphens, apostrophes
        
        return cleaned
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for similarity comparison"""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common suffixes for comparison
        suffixes = ['jr', 'jr.', 'sr', 'sr.', 'iii', 'ii', 'iv', 'v']
        for suffix in suffixes:
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(' ' + suffix)]
        
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _standardize_position(self, position: str) -> str:
        """Standardize position abbreviations"""
        if not position or pd.isna(position):
            return "UNK"
        
        pos = str(position).upper().strip()
        
        # Common position mappings
        position_mappings = {
            'QUARTERBACK': 'QB',
            'RUNNING BACK': 'RB', 'RUNNINGBACK': 'RB', 'HALFBACK': 'RB', 'FULLBACK': 'FB',
            'WIDE RECEIVER': 'WR', 'WIDERECEIVER': 'WR', 'RECEIVER': 'WR',
            'TIGHT END': 'TE', 'TIGHTEND': 'TE',
            'CENTER': 'C',
            'GUARD': 'G', 'LEFT GUARD': 'LG', 'RIGHT GUARD': 'RG',
            'TACKLE': 'T', 'LEFT TACKLE': 'LT', 'RIGHT TACKLE': 'RT',
            'DEFENSIVE END': 'DE', 'DEFENSIVEEND': 'DE',
            'DEFENSIVE TACKLE': 'DT', 'DEFENSIVETACKLE': 'DT',
            'NOSE TACKLE': 'NT', 'NOSETACKLE': 'NT',
            'LINEBACKER': 'LB', 'INSIDE LINEBACKER': 'ILB', 'OUTSIDE LINEBACKER': 'OLB',
            'MIDDLE LINEBACKER': 'MLB',
            'CORNERBACK': 'CB', 'CORNER': 'CB',
            'SAFETY': 'S', 'FREE SAFETY': 'FS', 'STRONG SAFETY': 'SS',
            'KICKER': 'K', 'PLACEKICKER': 'K',
            'PUNTER': 'P',
            'LONG SNAPPER': 'LS', 'LONGSNAPPER': 'LS'
        }
        
        return position_mappings.get(pos, pos)
    
    def _identify_identity_conflicts(self, player_data: pd.DataFrame) -> List[PlayerIdentityConflict]:
        """Identify potential player identity conflicts"""
        
        conflicts = []
        
        # Group by normalized names to find potential duplicates
        name_groups = defaultdict(list)
        
        for _, row in player_data.iterrows():
            normalized_name = row.get('name_normalized', '')
            if normalized_name:
                name_groups[normalized_name].append(row)
        
        # Analyze each name group for conflicts
        for normalized_name, group in name_groups.items():
            if len(group) > 1:
                # Check if multiple player IDs exist for this name
                unique_ids = list(set([str(row.get('player_id', '')) for row in group if row.get('player_id')]))
                
                if len(unique_ids) > 1:
                    # Potential conflict - analyze further
                    conflict_confidence = self._analyze_group_for_conflicts(group)
                    
                    if conflict_confidence > 0.3:  # Threshold for considering as potential conflict
                        conflict = PlayerIdentityConflict(
                            conflicting_ids=unique_ids,
                            player_names=[str(row.get('player_name', '')) for row in group],
                            positions=[str(row.get('position', '')) for row in group],
                            teams=[str(row.get('recent_team', '')) for row in group],
                            seasons=[int(row.get('season', 0)) for row in group if row.get('season')],
                            confidence_score=conflict_confidence,
                            resolution_method='pending'
                        )
                        conflicts.append(conflict)
        
        # Also check for similar (but not identical) names that might be the same player
        conflicts.extend(self._find_similar_name_conflicts(player_data))
        
        return conflicts
    
    def _analyze_group_for_conflicts(self, group: List[pd.Series]) -> float:
        """Analyze a group of records to determine if they represent the same player"""
        
        if len(group) < 2:
            return 0.0
        
        confidence_factors = []
        
        # 1. Name similarity analysis
        names = [str(row.get('player_name', '')) for row in group]
        name_similarities = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                similarity = self._calculate_name_similarity(names[i], names[j])
                name_similarities.append(similarity)
        
        if name_similarities:
            avg_name_similarity = np.mean(name_similarities)
            confidence_factors.append(avg_name_similarity * self.name_weight)
        
        # 2. Position consistency
        positions = [str(row.get('position', '')) for row in group if row.get('position')]
        if positions:
            position_consistency = self._calculate_position_consistency(positions)
            confidence_factors.append(position_consistency * self.position_weight)
        
        # 3. Team history overlap
        teams = [str(row.get('recent_team', '')) for row in group if row.get('recent_team')]
        seasons = [int(row.get('season', 0)) for row in group if row.get('season')]
        
        if teams and seasons:
            team_consistency = self._calculate_team_consistency(teams, seasons)
            confidence_factors.append(team_consistency * self.team_weight)
        
        # 4. Temporal overlap
        if len(seasons) > 1:
            temporal_consistency = self._calculate_temporal_consistency(seasons)
            confidence_factors.append(temporal_consistency * self.temporal_weight)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two player names"""
        if not name1 or not name2:
            return 0.0
        
        # Normalize names for comparison
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)
        
        if norm1 == norm2:
            return 1.0
        
        # Use sequence matcher for basic similarity
        basic_similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        # Check for common name patterns
        name1_parts = norm1.split()
        name2_parts = norm2.split()
        
        if len(name1_parts) >= 2 and len(name2_parts) >= 2:
            # Check if first and last names match
            first_match = name1_parts[0] == name2_parts[0]
            last_match = name1_parts[-1] == name2_parts[-1]
            
            if first_match and last_match:
                return 0.95  # Very high confidence for first+last match
            elif last_match:
                return 0.8   # High confidence for last name match
            elif first_match:
                return 0.6   # Moderate confidence for first name match
        
        # Check for nickname/abbreviation patterns
        if self._check_nickname_patterns(norm1, norm2):
            return 0.85
        
        return basic_similarity
    
    def _check_nickname_patterns(self, name1: str, name2: str) -> bool:
        """Check for common nickname patterns"""
        
        # Common nickname mappings
        nicknames = {
            'william': ['bill', 'billy', 'will'],
            'robert': ['bob', 'bobby', 'rob', 'robby'],
            'richard': ['rick', 'ricky', 'dick'],
            'michael': ['mike', 'mickey'],
            'christopher': ['chris'],
            'anthony': ['tony'],
            'benjamin': ['ben', 'benny'],
            'alexander': ['alex'],
            'jonathan': ['jon', 'johnny'],
            'matthew': ['matt'],
            'andrew': ['andy', 'drew'],
            'joshua': ['josh'],
            'daniel': ['dan', 'danny'],
            'david': ['dave', 'davey'],
            'james': ['jim', 'jimmy', 'jamie'],
            'john': ['johnny', 'jack'],
            'thomas': ['tom', 'tommy']
        }
        
        name1_parts = name1.split()
        name2_parts = name2.split()
        
        for part1 in name1_parts:
            for part2 in name2_parts:
                # Check if one is a nickname of the other
                for full_name, nicks in nicknames.items():
                    if (part1 == full_name and part2 in nicks) or (part2 == full_name and part1 in nicks):
                        return True
                    if part1 in nicks and part2 in nicks:  # Both are nicknames of same name
                        return True
        
        return False
    
    def _calculate_position_consistency(self, positions: List[str]) -> float:
        """Calculate consistency of positions"""
        
        unique_positions = set(pos for pos in positions if pos and pos != 'UNK')
        
        if len(unique_positions) <= 1:
            return 1.0  # All same position or no position data
        
        # Check if positions are in the same group (e.g., all skill positions)
        for group_name, group_positions in self.position_groups.items():
            if all(pos in group_positions for pos in unique_positions):
                return 0.8  # Same position group
        
        # Different position groups - less likely to be same player
        return 0.3
    
    def _calculate_team_consistency(self, teams: List[str], seasons: List[int]) -> float:
        """Calculate team consistency considering player movement"""
        
        unique_teams = set(team for team in teams if team)
        
        if len(unique_teams) <= 1:
            return 1.0  # Same team
        
        if len(unique_teams) <= 3:
            return 0.8  # Reasonable number of team changes
        
        if len(unique_teams) <= 5:
            return 0.5  # Many team changes but possible
        
        return 0.2  # Too many teams - unlikely same player
    
    def _calculate_temporal_consistency(self, seasons: List[int]) -> float:
        """Calculate temporal consistency of career"""
        
        if len(seasons) < 2:
            return 1.0
        
        season_range = max(seasons) - min(seasons)
        
        if season_range <= 15:  # Reasonable career length
            return 1.0
        elif season_range <= 20:  # Long but possible career
            return 0.7
        else:  # Unreasonably long career
            return 0.2
    
    def _find_similar_name_conflicts(self, player_data: pd.DataFrame) -> List[PlayerIdentityConflict]:
        """Find conflicts between similar (but not identical) names"""
        
        conflicts = []
        processed_pairs = set()
        
        # Get unique player records
        unique_players = player_data.drop_duplicates(subset=['player_id']).copy()
        
        for i, row1 in unique_players.iterrows():
            for j, row2 in unique_players.iterrows():
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                pair_key = tuple(sorted([row1['player_id'], row2['player_id']]))
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # Calculate similarity
                name_similarity = self._calculate_name_similarity(
                    row1.get('player_name', ''), 
                    row2.get('player_name', '')
                )
                
                # If names are similar but not identical, investigate further
                if 0.7 <= name_similarity < 1.0:
                    # Get all records for both players
                    player1_records = player_data[player_data['player_id'] == row1['player_id']]
                    player2_records = player_data[player_data['player_id'] == row2['player_id']]
                    
                    combined_records = pd.concat([player1_records, player2_records])
                    
                    conflict_confidence = self._analyze_group_for_conflicts(
                        [row for _, row in combined_records.iterrows()]
                    )
                    
                    if conflict_confidence > 0.6:  # High confidence these are the same player
                        conflict = PlayerIdentityConflict(
                            conflicting_ids=[str(row1['player_id']), str(row2['player_id'])],
                            player_names=[str(row1.get('player_name', '')), str(row2.get('player_name', ''))],
                            positions=[str(row1.get('position', '')), str(row2.get('position', ''))],
                            teams=[str(row1.get('recent_team', '')), str(row2.get('recent_team', ''))],
                            seasons=[int(row1.get('season', 0)), int(row2.get('season', 0))],
                            confidence_score=conflict_confidence,
                            resolution_method='similar_names'
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _resolve_identity_conflicts(self, conflicts: List[PlayerIdentityConflict], 
                                   player_data: pd.DataFrame) -> List[PlayerMasterRecord]:
        """Resolve identity conflicts and create master records"""
        
        master_records = []
        processed_ids = set()
        
        # Sort conflicts by confidence score (highest first)
        conflicts.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Process high-confidence conflicts first
        for conflict in conflicts:
            if conflict.confidence_score >= 0.8:  # High confidence merge
                if not any(pid in processed_ids for pid in conflict.conflicting_ids):
                    master_record = self._create_master_record_from_conflict(conflict, player_data)
                    master_records.append(master_record)
                    processed_ids.update(conflict.conflicting_ids)
                    
                    logger.debug(f"Merged high-confidence conflict: {conflict.conflicting_ids} -> {master_record.master_id}")
        
        # Process medium-confidence conflicts with additional validation
        for conflict in conflicts:
            if 0.6 <= conflict.confidence_score < 0.8:
                if not any(pid in processed_ids for pid in conflict.conflicting_ids):
                    # Additional validation for medium confidence
                    if self._validate_conflict_resolution(conflict, player_data):
                        master_record = self._create_master_record_from_conflict(conflict, player_data)
                        master_records.append(master_record)
                        processed_ids.update(conflict.conflicting_ids)
                        
                        logger.debug(f"Merged medium-confidence conflict: {conflict.conflicting_ids} -> {master_record.master_id}")
                    else:
                        # Keep as separate players
                        for player_id in conflict.conflicting_ids:
                            if player_id not in processed_ids:
                                master_record = self._create_master_record_from_id(player_id, player_data)
                                if master_record:
                                    master_records.append(master_record)
                                    processed_ids.add(player_id)
        
        # Keep low-confidence conflicts as separate players
        for conflict in conflicts:
            if conflict.confidence_score < 0.6:
                for player_id in conflict.conflicting_ids:
                    if player_id not in processed_ids:
                        master_record = self._create_master_record_from_id(player_id, player_data)
                        if master_record:
                            master_records.append(master_record)
                            processed_ids.add(player_id)
        
        # Process remaining players not involved in conflicts
        all_ids = set(str(pid) for pid in player_data['player_id'].dropna().unique())
        remaining_ids = all_ids - processed_ids
        
        for player_id in remaining_ids:
            master_record = self._create_master_record_from_id(player_id, player_data)
            if master_record:
                master_records.append(master_record)
        
        return master_records
    
    def _validate_conflict_resolution(self, conflict: PlayerIdentityConflict, 
                                    player_data: pd.DataFrame) -> bool:
        """Additional validation for medium-confidence conflicts"""
        
        # Get all records for conflicted players
        all_records = player_data[player_data['player_id'].isin(conflict.conflicting_ids)]
        
        # Check for temporal overlap (players can't play simultaneously)
        seasons_by_id = {}
        for _, row in all_records.iterrows():
            player_id = str(row['player_id'])
            season = row.get('season')
            if season:
                if player_id not in seasons_by_id:
                    seasons_by_id[player_id] = []
                seasons_by_id[player_id].append(int(season))
        
        # Check if there's any season overlap
        if len(seasons_by_id) > 1:
            all_seasons = [set(seasons) for seasons in seasons_by_id.values()]
            for i in range(len(all_seasons)):
                for j in range(i + 1, len(all_seasons)):
                    if all_seasons[i] & all_seasons[j]:  # Intersection exists
                        return False  # Same season overlap - likely different players
        
        return True
    
    def _create_master_record_from_conflict(self, conflict: PlayerIdentityConflict, 
                                          player_data: pd.DataFrame) -> PlayerMasterRecord:
        """Create master record by merging conflicted players"""
        
        # Get all records for conflicted IDs
        all_records = player_data[player_data['player_id'].isin(conflict.conflicting_ids)]
        
        # Choose canonical name (most recent or most complete)
        names = [str(name) for name in all_records['player_name'].dropna().tolist()]
        canonical_name = max(set(names), key=lambda x: (names.count(x), len(x))) if names else "Unknown"
        
        # Choose primary position (most common)
        positions = [str(pos) for pos in all_records['position'].dropna().tolist()]
        primary_position = max(set(positions), key=positions.count) if positions else "UNK"
        
        # Create master ID (use most recent or lexicographically first)
        master_id = sorted(conflict.conflicting_ids)[0]
        
        # Collect all data
        all_names = list(set([str(name) for name in all_records['player_name'].dropna().tolist()]))
        seasons_active = sorted([int(s) for s in all_records['season'].dropna().unique().tolist()])
        teams = list(set([str(team) for team in all_records['recent_team'].dropna().tolist()]))
        
        return PlayerMasterRecord(
            master_id=master_id,
            canonical_name=canonical_name,
            primary_position=primary_position,
            all_ids=conflict.conflicting_ids,
            all_names=all_names,
            seasons_active=seasons_active,
            teams_played_for=teams,
            confidence_score=conflict.confidence_score
        )
    
    def _create_master_record_from_id(self, player_id: str, 
                                     player_data: pd.DataFrame) -> Optional[PlayerMasterRecord]:
        """Create master record for a single player ID"""
        
        player_records = player_data[player_data['player_id'] == player_id]
        
        if player_records.empty:
            return None
        
        # Get player info
        name = str(player_records['player_name'].dropna().iloc[0]) if not player_records['player_name'].dropna().empty else "Unknown"
        position = str(player_records['position'].dropna().iloc[0]) if not player_records['position'].dropna().empty else "UNK"
        
        seasons = sorted([int(s) for s in player_records['season'].dropna().unique().tolist()])
        teams = list(set([str(team) for team in player_records['recent_team'].dropna().tolist()]))
        
        return PlayerMasterRecord(
            master_id=player_id,
            canonical_name=name,
            primary_position=position,
            all_ids=[player_id],
            all_names=[name],
            seasons_active=seasons,
            teams_played_for=teams,
            confidence_score=1.0  # Single ID, no conflict
        )
    
    def _create_id_mapping(self, master_records: List[PlayerMasterRecord]) -> Dict[str, str]:
        """Create mapping from all IDs to master IDs"""
        
        mapping = {}
        
        for record in master_records:
            for player_id in record.all_ids:
                mapping[str(player_id)] = record.master_id
        
        return mapping
    
    def _validate_mapping_quality(self, mapping: Dict[str, str], 
                                 player_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of the identity mapping"""
        
        total_ids = len(set(str(pid) for pid in player_data['player_id'].dropna().unique()))
        mapped_ids = len(mapping)
        unique_master_ids = len(set(mapping.values()))
        
        # Calculate consolidation ratio
        consolidation_ratio = 1 - (unique_master_ids / total_ids) if total_ids > 0 else 0
        
        # Check for reasonable consolidation (not too much, not too little)
        if 0.01 <= consolidation_ratio <= 0.15:  # 1-15% consolidation is reasonable
            quality_score = 0.9
        elif 0.15 < consolidation_ratio <= 0.25:  # Higher consolidation, still acceptable
            quality_score = 0.7
        elif consolidation_ratio > 0.25:  # Too much consolidation - likely over-merging
            quality_score = 0.5
        else:  # Very little consolidation - likely under-merging
            quality_score = 0.8
        
        return {
            'total_original_ids': total_ids,
            'mapped_ids': mapped_ids,
            'unique_master_ids': unique_master_ids,
            'consolidation_ratio': consolidation_ratio,
            'quality_score': quality_score,
            'coverage': mapped_ids / total_ids if total_ids > 0 else 0
        }
