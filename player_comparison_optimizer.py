"""
Player Comparison and Lineup Optimization for NFL Betting
Advanced analytics for comparing players and optimizing DFS lineups
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from itertools import combinations
import pulp
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PlayerComparison:
    player1: str
    player2: str
    similarity_score: float
    stat_differences: Dict[str, float]
    recommendation: str

@dataclass
class LineupPlayer:
    player_id: str
    position: str
    salary: int
    projected_points: float
    ownership_projection: float
    value_score: float

@dataclass
class OptimalLineup:
    players: List[LineupPlayer]
    total_salary: int
    projected_points: float
    expected_ownership: float
    lineup_type: str  # 'cash', 'gpp', 'balanced'

class PlayerComparator:
    """Advanced player comparison and analysis."""
    
    def __init__(self, database_path: str = "data/nfl_predictions.db"):
        self.db_path = database_path
        self.comparison_cache = {}
    
    def compare_players(self, player1: str, player2: str, 
                       stats_to_compare: List[str] = None) -> PlayerComparison:
        """Compare two players across multiple statistical categories."""
        
        if stats_to_compare is None:
            stats_to_compare = [
                'fantasy_points_ppr', 'passing_yards', 'rushing_yards', 
                'receiving_yards', 'touchdowns', 'receptions'
            ]
        
        # Get historical data for both players
        player1_data = self._get_player_historical_stats(player1)
        player2_data = self._get_player_historical_stats(player2)
        
        if not player1_data or not player2_data:
            return PlayerComparison(
                player1=player1,
                player2=player2,
                similarity_score=0.0,
                stat_differences={},
                recommendation="Insufficient data for comparison"
            )
        
        # Calculate averages for comparison stats
        player1_avgs = self._calculate_stat_averages(player1_data, stats_to_compare)
        player2_avgs = self._calculate_stat_averages(player2_data, stats_to_compare)
        
        # Calculate similarity score using cosine similarity
        p1_vector = np.array([player1_avgs.get(stat, 0) for stat in stats_to_compare])
        p2_vector = np.array([player2_avgs.get(stat, 0) for stat in stats_to_compare])
        
        # Normalize vectors
        scaler = StandardScaler()
        vectors = scaler.fit_transform([p1_vector, p2_vector])
        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        
        # Calculate stat differences
        stat_differences = {}
        for stat in stats_to_compare:
            diff = player1_avgs.get(stat, 0) - player2_avgs.get(stat, 0)
            stat_differences[stat] = diff
        
        # Generate recommendation
        recommendation = self._generate_comparison_recommendation(
            player1, player2, similarity, stat_differences
        )
        
        return PlayerComparison(
            player1=player1,
            player2=player2,
            similarity_score=similarity,
            stat_differences=stat_differences,
            recommendation=recommendation
        )
    
    def find_similar_players(self, target_player: str, position: str = None, 
                           top_n: int = 5) -> List[PlayerComparison]:
        """Find players most similar to the target player."""
        
        # Get all players in the same position or all players
        candidate_players = self._get_candidate_players(position)
        
        # Remove target player from candidates
        candidate_players = [p for p in candidate_players if p != target_player]
        
        similarities = []
        for candidate in candidate_players:
            comparison = self.compare_players(target_player, candidate)
            if comparison.similarity_score > 0:
                similarities.append(comparison)
        
        # Sort by similarity score and return top N
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_n]
    
    def create_comparison_matrix(self, players: List[str]) -> pd.DataFrame:
        """Create a similarity matrix for multiple players."""
        
        n_players = len(players)
        similarity_matrix = np.zeros((n_players, n_players))
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                elif i < j:  # Only calculate upper triangle
                    comparison = self.compare_players(player1, player2)
                    similarity_matrix[i][j] = comparison.similarity_score
                    similarity_matrix[j][i] = comparison.similarity_score  # Mirror
        
        return pd.DataFrame(similarity_matrix, index=players, columns=players)
    
    def _get_player_historical_stats(self, player_id: str) -> List[Dict]:
        """Get historical statistics for a player."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM player_game_stats 
            WHERE player_id = ? 
            ORDER BY created_at DESC 
            LIMIT 20
        """
        
        cursor.execute(query, (player_id,))
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def _calculate_stat_averages(self, player_data: List[Dict], stats: List[str]) -> Dict[str, float]:
        """Calculate average statistics for a player."""
        averages = {}
        
        for stat in stats:
            values = [game.get(stat, 0) or 0 for game in player_data]
            averages[stat] = np.mean(values) if values else 0
        
        return averages
    
    def _get_candidate_players(self, position: str = None) -> List[str]:
        """Get list of candidate players for comparison."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if position:
            query = """
                SELECT DISTINCT player_id FROM player_game_stats 
                WHERE player_id LIKE ?
                AND fantasy_points_ppr > 5
            """
            cursor.execute(query, (f"%_{position.lower()}",))
        else:
            query = """
                SELECT DISTINCT player_id FROM player_game_stats 
                WHERE fantasy_points_ppr > 5
            """
            cursor.execute(query)
        
        players = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return players
    
    def _generate_comparison_recommendation(self, player1: str, player2: str, 
                                         similarity: float, differences: Dict) -> str:
        """Generate recommendation based on comparison results."""
        
        if similarity > 0.8:
            return f"Very similar players - consider as alternatives in lineups"
        elif similarity > 0.6:
            return f"Moderately similar - {player1} may be better value if cheaper"
        elif similarity > 0.4:
            return f"Different playing styles - analyze matchups individually"
        else:
            return f"Very different players - not suitable for direct comparison"


class LineupOptimizer:
    """Optimize DFS lineups using advanced algorithms."""
    
    def __init__(self):
        self.position_requirements = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'FLEX': 1,  # RB/WR/TE
            'DST': 1
        }
        self.salary_cap = 50000  # DraftKings salary cap
    
    def optimize_lineup(self, players: List[LineupPlayer], 
                       lineup_type: str = 'balanced') -> OptimalLineup:
        """Optimize lineup using linear programming."""
        
        # Create optimization problem
        prob = pulp.LpProblem("DFS_Lineup_Optimization", pulp.LpMaximize)
        
        # Decision variables - binary for each player
        player_vars = {}
        for i, player in enumerate(players):
            player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
        
        # Objective function - maximize projected points
        if lineup_type == 'cash':
            # Cash games: maximize floor (conservative projections)
            prob += pulp.lpSum([
                player_vars[i] * (player.projected_points * 0.8)  # Conservative estimate
                for i, player in enumerate(players)
            ])
        elif lineup_type == 'gpp':
            # GPP: maximize ceiling with ownership consideration
            prob += pulp.lpSum([
                player_vars[i] * (player.projected_points * (1 + (1 - player.ownership_projection)))
                for i, player in enumerate(players)
            ])
        else:  # balanced
            prob += pulp.lpSum([
                player_vars[i] * player.projected_points
                for i, player in enumerate(players)
            ])
        
        # Salary constraint
        prob += pulp.lpSum([
            player_vars[i] * player.salary
            for i, player in enumerate(players)
        ]) <= self.salary_cap
        
        # Position constraints
        for position, required in self.position_requirements.items():
            if position == 'FLEX':
                # FLEX can be RB, WR, or TE
                prob += pulp.lpSum([
                    player_vars[i]
                    for i, player in enumerate(players)
                    if player.position in ['RB', 'WR', 'TE']
                ]) >= required + sum([self.position_requirements[pos] 
                                    for pos in ['RB', 'WR', 'TE']])
            else:
                prob += pulp.lpSum([
                    player_vars[i]
                    for i, player in enumerate(players)
                    if player.position == position
                ]) == required
        
        # Total players constraint
        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 9
        
        # Solve optimization
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        selected_players = []
        total_salary = 0
        projected_points = 0
        expected_ownership = 0
        
        for i, player in enumerate(players):
            if player_vars[i].value() == 1:
                selected_players.append(player)
                total_salary += player.salary
                projected_points += player.projected_points
                expected_ownership += player.ownership_projection
        
        expected_ownership /= len(selected_players)
        
        return OptimalLineup(
            players=selected_players,
            total_salary=total_salary,
            projected_points=projected_points,
            expected_ownership=expected_ownership,
            lineup_type=lineup_type
        )
    
    def generate_multiple_lineups(self, players: List[LineupPlayer], 
                                num_lineups: int = 5, 
                                lineup_type: str = 'balanced') -> List[OptimalLineup]:
        """Generate multiple optimal lineups with player diversity."""
        
        lineups = []
        used_players = set()
        
        for i in range(num_lineups):
            # Filter out overused players for diversity
            available_players = [
                p for p in players 
                if used_players.count(p.player_id) < 2  # Max 2 appearances
            ]
            
            if len(available_players) < 20:  # Need minimum players
                available_players = players  # Reset if too few
            
            lineup = self.optimize_lineup(available_players, lineup_type)
            lineups.append(lineup)
            
            # Track used players
            for player in lineup.players:
                used_players.add(player.player_id)
        
        return lineups
    
    def calculate_lineup_correlation(self, lineup: OptimalLineup) -> float:
        """Calculate correlation risk in lineup (game stacking, etc.)."""
        
        # Simple correlation based on same team players
        teams = {}
        for player in lineup.players:
            team = player.player_id.split('_')[-1] if '_' in player.player_id else 'UNK'
            teams[team] = teams.get(team, 0) + 1
        
        # High correlation if many players from same team
        max_team_players = max(teams.values())
        correlation_score = max_team_players / len(lineup.players)
        
        return correlation_score


class AdvancedAnalytics:
    """Advanced analytics for player and lineup analysis."""
    
    def __init__(self):
        self.analytics_cache = {}
    
    def calculate_player_consistency(self, player_data: List[Dict], 
                                   stat: str = 'fantasy_points_ppr') -> Dict:
        """Calculate consistency metrics for a player."""
        
        values = [game.get(stat, 0) or 0 for game in player_data]
        
        if not values:
            return {}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Coefficient of variation (lower is more consistent)
        cv = std_val / mean_val if mean_val > 0 else float('inf')
        
        # Floor and ceiling
        floor = np.percentile(values, 10)  # 10th percentile
        ceiling = np.percentile(values, 90)  # 90th percentile
        
        # Boom/bust rate
        boom_threshold = mean_val * 1.5
        bust_threshold = mean_val * 0.5
        
        boom_rate = sum(1 for v in values if v >= boom_threshold) / len(values)
        bust_rate = sum(1 for v in values if v <= bust_threshold) / len(values)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'coefficient_of_variation': cv,
            'floor': floor,
            'ceiling': ceiling,
            'boom_rate': boom_rate,
            'bust_rate': bust_rate,
            'consistency_grade': self._grade_consistency(cv)
        }
    
    def _grade_consistency(self, cv: float) -> str:
        """Grade consistency based on coefficient of variation."""
        if cv < 0.3:
            return 'A'  # Very consistent
        elif cv < 0.5:
            return 'B'  # Good consistency
        elif cv < 0.7:
            return 'C'  # Average consistency
        elif cv < 1.0:
            return 'D'  # Below average
        else:
            return 'F'  # Very inconsistent
    
    def analyze_matchup_history(self, player_id: str, opponent_team: str) -> Dict:
        """Analyze player's historical performance against specific opponent."""
        
        # This would require opponent data in the database
        # For now, return placeholder structure
        return {
            'games_played': 0,
            'avg_fantasy_points': 0,
            'best_game': 0,
            'worst_game': 0,
            'trend': 'NEUTRAL'
        }
    
    def calculate_value_metrics(self, player: LineupPlayer) -> Dict:
        """Calculate various value metrics for a player."""
        
        # Points per dollar
        ppd = player.projected_points / (player.salary / 1000) if player.salary > 0 else 0
        
        # Ownership adjusted value
        ownership_adj_value = player.projected_points * (1 - player.ownership_projection)
        
        # Value tier (relative to position)
        value_tier = self._determine_value_tier(ppd)
        
        return {
            'points_per_dollar': ppd,
            'ownership_adjusted_value': ownership_adj_value,
            'value_tier': value_tier,
            'salary_percentile': 0,  # Would need position salary data
            'projection_percentile': 0  # Would need position projection data
        }
    
    def _determine_value_tier(self, ppd: float) -> str:
        """Determine value tier based on points per dollar."""
        if ppd >= 3.0:
            return 'ELITE'
        elif ppd >= 2.5:
            return 'GREAT'
        elif ppd >= 2.0:
            return 'GOOD'
        elif ppd >= 1.5:
            return 'FAIR'
        else:
            return 'POOR'


class VisualizationEngine:
    """Create visualizations for player comparisons and lineup analysis."""
    
    def __init__(self, output_dir: Path = Path("visualizations")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_player_comparison(self, comparison: PlayerComparison, 
                             save_path: str = None) -> str:
        """Create visualization comparing two players."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stat differences bar chart
        stats = list(comparison.stat_differences.keys())
        differences = list(comparison.stat_differences.values())
        
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        ax1.barh(stats, differences, color=colors, alpha=0.7)
        ax1.set_title(f'{comparison.player1} vs {comparison.player2}\nStat Differences')
        ax1.set_xlabel('Difference (Player1 - Player2)')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Similarity score gauge
        similarity = comparison.similarity_score
        ax2.pie([similarity, 1-similarity], labels=['Similar', 'Different'], 
                colors=['lightgreen', 'lightcoral'], startangle=90)
        ax2.set_title(f'Similarity Score: {similarity:.2f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            save_path = self.output_dir / f"comparison_{comparison.player1}_{comparison.player2}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
    
    def plot_lineup_analysis(self, lineup: OptimalLineup, save_path: str = None) -> str:
        """Create visualization analyzing an optimal lineup."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Salary distribution by position
        positions = [p.position for p in lineup.players]
        salaries = [p.salary for p in lineup.players]
        
        position_salary = {}
        for pos, sal in zip(positions, salaries):
            position_salary[pos] = position_salary.get(pos, 0) + sal
        
        ax1.bar(position_salary.keys(), position_salary.values())
        ax1.set_title('Salary Distribution by Position')
        ax1.set_ylabel('Total Salary ($)')
        
        # Projected points by player
        players = [p.player_id.split('_')[0] for p in lineup.players]
        points = [p.projected_points for p in lineup.players]
        
        ax2.barh(players, points)
        ax2.set_title('Projected Points by Player')
        ax2.set_xlabel('Projected Points')
        
        # Value scatter plot
        values = [p.projected_points / (p.salary / 1000) for p in lineup.players]
        ownership = [p.ownership_projection for p in lineup.players]
        
        scatter = ax3.scatter(ownership, values, s=100, alpha=0.7)
        ax3.set_xlabel('Ownership Projection')
        ax3.set_ylabel('Points per $1K')
        ax3.set_title('Value vs Ownership')
        
        # Lineup summary
        ax4.axis('off')
        summary_text = f"""
        Lineup Summary:
        
        Total Salary: ${lineup.total_salary:,}
        Projected Points: {lineup.projected_points:.1f}
        Expected Ownership: {lineup.expected_ownership:.1%}
        Lineup Type: {lineup.lineup_type.upper()}
        
        Salary Remaining: ${50000 - lineup.total_salary:,}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            save_path = self.output_dir / f"lineup_analysis_{lineup.lineup_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)


# Example usage and testing
def main():
    """Example usage of player comparison and lineup optimization."""
    
    print("🔍 Player Comparison & Lineup Optimization Demo")
    print("=" * 60)
    
    # Initialize components
    comparator = PlayerComparator()
    optimizer = LineupOptimizer()
    analytics = AdvancedAnalytics()
    visualizer = VisualizationEngine()
    
    # Example player comparison
    print("\n📊 Player Comparison Example:")
    comparison = comparator.compare_players("pmahomes_qb", "jallen_qb")
    print(f"   Similarity Score: {comparison.similarity_score:.2f}")
    print(f"   Recommendation: {comparison.recommendation}")
    
    # Example lineup optimization
    print("\n🎯 Lineup Optimization Example:")
    
    # Create sample players (in real implementation, this would come from database)
    sample_players = [
        LineupPlayer("pmahomes_qb", "QB", 8000, 22.5, 0.25, 2.8),
        LineupPlayer("cmccaffrey_rb", "RB", 9500, 24.2, 0.30, 2.5),
        LineupPlayer("jchase_wr", "WR", 8500, 20.1, 0.20, 2.4),
        # Add more players...
    ]
    
    if len(sample_players) >= 9:  # Need minimum for lineup
        optimal_lineup = optimizer.optimize_lineup(sample_players, 'balanced')
        print(f"   Projected Points: {optimal_lineup.projected_points:.1f}")
        print(f"   Total Salary: ${optimal_lineup.total_salary:,}")
        print(f"   Players: {len(optimal_lineup.players)}")
    
    print("\n✅ Player comparison and optimization tools ready!")


if __name__ == "__main__":
    main()
