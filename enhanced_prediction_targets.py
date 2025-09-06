"""
Enhanced Prediction Targets for Comprehensive NFL Statistics
Defines all major NFL statistics for prediction across all positions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class StatCategory(Enum):
    """Categories of NFL statistics."""
    PASSING = "passing"
    RUSHING = "rushing"
    RECEIVING = "receiving"
    DEFENSE = "defense"
    SPECIAL_TEAMS = "special_teams"
    FANTASY = "fantasy"

@dataclass
class PredictionTarget:
    """Enhanced prediction target with comprehensive metadata."""
    name: str
    column: str
    category: StatCategory
    positions: List[str]
    prediction_type: str  # 'regression' or 'classification'
    min_value: float = 0.0
    max_value: Optional[float] = None
    is_prop_bet: bool = True
    confidence_threshold: float = 0.6
    description: str = ""
    
class EnhancedPredictionTargets:
    """Comprehensive prediction targets for all NFL positions and statistics."""
    
    def __init__(self):
        """Initialize all prediction targets."""
        self.targets = self._initialize_all_targets()
        
    def _initialize_all_targets(self) -> Dict[str, List[PredictionTarget]]:
        """Initialize comprehensive prediction targets for all positions."""
        
        # Passing Statistics
        passing_targets = [
            PredictionTarget(
                name="passing_attempts", column="passing_attempts", 
                category=StatCategory.PASSING, positions=["QB"],
                prediction_type="regression", min_value=0, max_value=70,
                description="Number of pass attempts"
            ),
            PredictionTarget(
                name="passing_completions", column="passing_completions",
                category=StatCategory.PASSING, positions=["QB"],
                prediction_type="regression", min_value=0, max_value=50,
                description="Number of completed passes"
            ),
            PredictionTarget(
                name="passing_yards", column="passing_yards",
                category=StatCategory.PASSING, positions=["QB"],
                prediction_type="regression", min_value=0, max_value=500,
                description="Total passing yards"
            ),
            PredictionTarget(
                name="passing_touchdowns", column="passing_touchdowns",
                category=StatCategory.PASSING, positions=["QB"],
                prediction_type="regression", min_value=0, max_value=7,
                description="Passing touchdowns"
            ),
            PredictionTarget(
                name="passing_interceptions", column="passing_interceptions",
                category=StatCategory.PASSING, positions=["QB"],
                prediction_type="regression", min_value=0, max_value=5,
                description="Interceptions thrown"
            ),
            PredictionTarget(
                name="passing_sacks", column="passing_sacks",
                category=StatCategory.PASSING, positions=["QB"],
                prediction_type="regression", min_value=0, max_value=10,
                description="Times sacked"
            ),
            PredictionTarget(
                name="passing_sack_yards", column="passing_sack_yards",
                category=StatCategory.PASSING, positions=["QB"],
                prediction_type="regression", min_value=0, max_value=100,
                description="Yards lost to sacks"
            ),
        ]
        
        # Rushing Statistics
        rushing_targets = [
            PredictionTarget(
                name="rushing_attempts", column="rushing_attempts",
                category=StatCategory.RUSHING, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=35,
                description="Number of rushing attempts"
            ),
            PredictionTarget(
                name="rushing_yards", column="rushing_yards",
                category=StatCategory.RUSHING, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=300,
                description="Total rushing yards"
            ),
            PredictionTarget(
                name="rushing_touchdowns", column="rushing_touchdowns",
                category=StatCategory.RUSHING, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=4,
                description="Rushing touchdowns"
            ),
            PredictionTarget(
                name="rushing_fumbles", column="rushing_fumbles",
                category=StatCategory.RUSHING, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=3,
                description="Rushing fumbles"
            ),
            PredictionTarget(
                name="rushing_first_downs", column="rushing_first_downs",
                category=StatCategory.RUSHING, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=15,
                description="First downs via rushing"
            ),
        ]
        
        # Receiving Statistics
        receiving_targets = [
            PredictionTarget(
                name="targets", column="targets",
                category=StatCategory.RECEIVING, positions=["RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=25,
                description="Number of targets"
            ),
            PredictionTarget(
                name="receptions", column="receptions",
                category=StatCategory.RECEIVING, positions=["RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=20,
                description="Number of receptions"
            ),
            PredictionTarget(
                name="receiving_yards", column="receiving_yards",
                category=StatCategory.RECEIVING, positions=["RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=250,
                description="Total receiving yards"
            ),
            PredictionTarget(
                name="receiving_touchdowns", column="receiving_touchdowns",
                category=StatCategory.RECEIVING, positions=["RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=4,
                description="Receiving touchdowns"
            ),
            PredictionTarget(
                name="receiving_fumbles", column="receiving_fumbles",
                category=StatCategory.RECEIVING, positions=["RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=2,
                description="Receiving fumbles"
            ),
            PredictionTarget(
                name="receiving_first_downs", column="receiving_first_downs",
                category=StatCategory.RECEIVING, positions=["RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=12,
                description="First downs via receiving"
            ),
        ]
        
        # Fantasy Points
        fantasy_targets = [
            PredictionTarget(
                name="fantasy_points_standard", column="fantasy_points_standard",
                category=StatCategory.FANTASY, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=50,
                description="Standard fantasy points"
            ),
            PredictionTarget(
                name="fantasy_points_ppr", column="fantasy_points_ppr",
                category=StatCategory.FANTASY, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=50,
                description="PPR fantasy points"
            ),
            PredictionTarget(
                name="fantasy_points_half_ppr", column="fantasy_points_half_ppr",
                category=StatCategory.FANTASY, positions=["QB", "RB", "WR", "TE"],
                prediction_type="regression", min_value=0, max_value=50,
                description="Half-PPR fantasy points"
            ),
        ]
        
        # Organize by position
        position_targets = {
            'QB': [],
            'RB': [],
            'WR': [],
            'TE': [],
            'DEF': [],  # Defense (team-level)
            'K': []     # Kicker
        }
        
        # Add all targets to appropriate positions
        all_targets = passing_targets + rushing_targets + receiving_targets + fantasy_targets
        
        for target in all_targets:
            for position in target.positions:
                if position in position_targets:
                    position_targets[position].append(target)
        
        # Add defense-specific targets (would be team-level predictions)
        defense_targets = [
            PredictionTarget(
                name="def_tackles", column="def_tackles",
                category=StatCategory.DEFENSE, positions=["DEF"],
                prediction_type="regression", min_value=0, max_value=15,
                description="Total tackles"
            ),
            PredictionTarget(
                name="def_assists", column="def_assists",
                category=StatCategory.DEFENSE, positions=["DEF"],
                prediction_type="regression", min_value=0, max_value=10,
                description="Tackle assists"
            ),
            PredictionTarget(
                name="def_sacks", column="def_sacks",
                category=StatCategory.DEFENSE, positions=["DEF"],
                prediction_type="regression", min_value=0, max_value=5,
                description="Sacks recorded"
            ),
            PredictionTarget(
                name="def_interceptions", column="def_interceptions",
                category=StatCategory.DEFENSE, positions=["DEF"],
                prediction_type="regression", min_value=0, max_value=3,
                description="Interceptions"
            ),
            PredictionTarget(
                name="def_pass_deflections", column="def_pass_deflections",
                category=StatCategory.DEFENSE, positions=["DEF"],
                prediction_type="regression", min_value=0, max_value=8,
                description="Pass deflections"
            ),
        ]
        
        position_targets['DEF'].extend(defense_targets)
        
        # Add kicker-specific targets
        kicker_targets = [
            PredictionTarget(
                name="fg_made", column="fg_made",
                category=StatCategory.SPECIAL_TEAMS, positions=["K"],
                prediction_type="regression", min_value=0, max_value=6,
                description="Field goals made"
            ),
            PredictionTarget(
                name="fg_attempted", column="fg_attempted",
                category=StatCategory.SPECIAL_TEAMS, positions=["K"],
                prediction_type="regression", min_value=0, max_value=8,
                description="Field goals attempted"
            ),
            PredictionTarget(
                name="extra_points_made", column="extra_points_made",
                category=StatCategory.SPECIAL_TEAMS, positions=["K"],
                prediction_type="regression", min_value=0, max_value=8,
                description="Extra points made"
            ),
            PredictionTarget(
                name="extra_points_attempted", column="extra_points_attempted",
                category=StatCategory.SPECIAL_TEAMS, positions=["K"],
                prediction_type="regression", min_value=0, max_value=8,
                description="Extra points attempted"
            ),
        ]
        
        position_targets['K'].extend(kicker_targets)
        
        return position_targets
    
    def get_targets_for_position(self, position: str) -> List[PredictionTarget]:
        """Get all prediction targets for a specific position."""
        return self.targets.get(position, [])
    
    def get_targets_by_category(self, category: StatCategory) -> List[PredictionTarget]:
        """Get all targets for a specific category."""
        targets = []
        for position_targets in self.targets.values():
            for target in position_targets:
                if target.category == category:
                    targets.append(target)
        return targets
    
    def get_prop_bet_targets(self) -> List[PredictionTarget]:
        """Get all targets suitable for prop betting."""
        targets = []
        for position_targets in self.targets.values():
            for target in position_targets:
                if target.is_prop_bet:
                    targets.append(target)
        return targets
    
    def get_target_by_name(self, name: str, position: str = None) -> Optional[PredictionTarget]:
        """Get a specific target by name and optionally position."""
        if position:
            position_targets = self.targets.get(position, [])
            for target in position_targets:
                if target.name == name:
                    return target
        else:
            # Search all positions
            for position_targets in self.targets.values():
                for target in position_targets:
                    if target.name == name:
                        return target
        return None
    
    def get_all_targets(self) -> List[PredictionTarget]:
        """Get all prediction targets across all positions."""
        all_targets = []
        for position_targets in self.targets.values():
            all_targets.extend(position_targets)
        return all_targets
    
    def get_position_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of targets by position and category."""
        summary = {}
        
        for position, targets in self.targets.items():
            summary[position] = {}
            for category in StatCategory:
                count = len([t for t in targets if t.category == category])
                if count > 0:
                    summary[position][category.value] = count
        
        return summary

# Global instance
PREDICTION_TARGETS = EnhancedPredictionTargets()

# Convenience functions
def get_targets_for_position(position: str) -> List[PredictionTarget]:
    """Get prediction targets for a position."""
    return PREDICTION_TARGETS.get_targets_for_position(position)

def get_prop_bet_targets() -> List[PredictionTarget]:
    """Get all prop bet targets."""
    return PREDICTION_TARGETS.get_prop_bet_targets()

def get_target_by_name(name: str, position: str = None) -> Optional[PredictionTarget]:
    """Get target by name."""
    return PREDICTION_TARGETS.get_target_by_name(name, position)

if __name__ == "__main__":
    # Example usage and testing
    targets = EnhancedPredictionTargets()
    
    print("NFL Prediction Targets Summary:")
    print("=" * 50)
    
    summary = targets.get_position_summary()
    for position, categories in summary.items():
        print(f"\n{position}:")
        for category, count in categories.items():
            print(f"  {category}: {count} targets")
    
    print(f"\nTotal prop bet targets: {len(targets.get_prop_bet_targets())}")
    print(f"Total targets across all positions: {len(targets.get_all_targets())}")
    
    # Show QB targets as example
    print(f"\nQB Targets:")
    qb_targets = targets.get_targets_for_position('QB')
    for target in qb_targets:
        print(f"  {target.name}: {target.description}")
