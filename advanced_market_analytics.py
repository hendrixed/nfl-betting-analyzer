"""
Advanced Market Analytics and Betting Intelligence
Sharp money detection, line movement patterns, public betting data, and value identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
import requests
import json
from collections import defaultdict
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class MarketIntelligence:
    player_id: str
    stat_type: str
    opening_line: float
    current_line: float
    line_movement: float
    sharp_money_indicator: str
    public_betting_percentage: float
    handle_percentage: float
    value_rating: str
    recommended_action: str

@dataclass
class BettingEdge:
    bet_type: str
    player_id: str
    edge_percentage: float
    confidence_level: str
    market_inefficiency: str
    recommended_stake: float

class AdvancedMarketAnalyzer:
    """Analyze betting markets for edges and inefficiencies."""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.market_cache = {}
        self._init_market_database()
        
        # Market analysis thresholds
        self.thresholds = {
            'sharp_money_line_move': 0.5,  # 0.5 point move indicates sharp money
            'reverse_line_movement': 0.3,   # Line moves opposite to public betting
            'steam_move': 1.0,              # 1+ point rapid movement
            'value_threshold': 0.05,        # 5% edge minimum for value bet
            'max_stake_percentage': 0.03    # Max 3% of bankroll per bet
        }
    
    def _init_market_database(self):
        """Initialize market data tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS line_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                stat_type TEXT,
                sportsbook TEXT,
                opening_line REAL,
                current_line REAL,
                line_movement REAL,
                timestamp DATETIME,
                public_percentage REAL,
                handle_percentage REAL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_line_movements_player_stat_time 
            ON line_movements(player_id, stat_type, timestamp)
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sharp_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                stat_type TEXT,
                indicator_type TEXT,
                strength REAL,
                timestamp DATETIME
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sharp_indicators_player_stat_time 
            ON sharp_indicators(player_id, stat_type, timestamp)
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS value_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                stat_type TEXT,
                predicted_value REAL,
                market_line REAL,
                edge_percentage REAL,
                confidence_score REAL,
                timestamp DATETIME,
                result TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_value_bets_player_time 
            ON value_bets(player_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def detect_sharp_money_movement(self, player_id: str, stat_type: str) -> Dict[str, Any]:
        """Detect sharp money indicators in line movement."""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get recent line movements
        query = """
            SELECT opening_line, current_line, line_movement, public_percentage, handle_percentage, timestamp
            FROM line_movements
            WHERE player_id = ? AND stat_type = ?
            AND timestamp > datetime('now', '-24 hours')
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(player_id, stat_type))
        conn.close()
        
        if df.empty:
            return {}
        
        latest = df.iloc[0]
        line_move = latest['line_movement']
        public_pct = latest['public_percentage'] or 50
        handle_pct = latest['handle_percentage'] or 50
        
        # Sharp money indicators
        indicators = []
        
        # Reverse line movement (line moves opposite to public)
        if line_move > self.thresholds['sharp_money_line_move'] and public_pct < 45:
            indicators.append("REVERSE_LINE_MOVEMENT_OVER")
        elif line_move < -self.thresholds['sharp_money_line_move'] and public_pct > 55:
            indicators.append("REVERSE_LINE_MOVEMENT_UNDER")
        
        # Handle vs ticket discrepancy
        if abs(handle_pct - public_pct) > 15:
            if handle_pct > public_pct:
                indicators.append("SHARP_MONEY_HANDLE_OVER")
            else:
                indicators.append("SHARP_MONEY_HANDLE_UNDER")
        
        # Steam move (rapid line movement)
        if abs(line_move) >= self.thresholds['steam_move']:
            indicators.append("STEAM_MOVE")
        
        # Calculate sharp money strength
        strength = 0
        if "REVERSE_LINE_MOVEMENT" in str(indicators):
            strength += 0.4
        if "SHARP_MONEY_HANDLE" in str(indicators):
            strength += 0.3
        if "STEAM_MOVE" in str(indicators):
            strength += 0.3
        
        return {
            'indicators': indicators,
            'strength': strength,
            'line_movement': line_move,
            'public_percentage': public_pct,
            'handle_percentage': handle_pct,
            'recommendation': self._generate_sharp_money_recommendation(indicators, strength)
        }
    
    def _generate_sharp_money_recommendation(self, indicators: List[str], strength: float) -> str:
        """Generate recommendation based on sharp money indicators."""
        
        if strength >= 0.7:
            return "STRONG_FOLLOW_SHARP_MONEY"
        elif strength >= 0.4:
            return "MODERATE_FOLLOW_SHARP_MONEY"
        elif strength >= 0.2:
            return "WEAK_SHARP_SIGNAL"
        else:
            return "NO_SHARP_SIGNAL"
    
    def calculate_market_value(self, player_id: str, stat_type: str, 
                             predicted_value: float, confidence: float) -> BettingEdge:
        """Calculate betting value and edge for a player prop."""
        
        # Get current market line
        market_line = self._get_current_market_line(player_id, stat_type)
        
        if not market_line:
            return BettingEdge(
                bet_type=f"{stat_type}_over",
                player_id=player_id,
                edge_percentage=0.0,
                confidence_level="INSUFFICIENT_DATA",
                market_inefficiency="NO_MARKET_DATA",
                recommended_stake=0.0
            )
        
        # Calculate edge
        edge_over = (predicted_value - market_line) / market_line
        edge_under = (market_line - predicted_value) / market_line
        
        # Determine best bet
        if edge_over > self.thresholds['value_threshold']:
            bet_type = f"{stat_type}_over"
            edge = edge_over
            market_inefficiency = "MARKET_UNDERVALUING"
        elif edge_under > self.thresholds['value_threshold']:
            bet_type = f"{stat_type}_under"
            edge = edge_under
            market_inefficiency = "MARKET_OVERVALUING"
        else:
            bet_type = "NO_BET"
            edge = 0.0
            market_inefficiency = "EFFICIENT_MARKET"
        
        # Confidence level
        if confidence >= 0.8 and edge >= 0.1:
            confidence_level = "HIGH"
        elif confidence >= 0.65 and edge >= 0.05:
            confidence_level = "MEDIUM"
        elif confidence >= 0.5 and edge >= 0.03:
            confidence_level = "LOW"
        else:
            confidence_level = "NO_CONFIDENCE"
        
        # Recommended stake (Kelly Criterion approximation)
        if edge > 0 and confidence > 0.5:
            kelly_fraction = (edge * confidence) / 1.0  # Simplified Kelly
            recommended_stake = min(kelly_fraction * self.thresholds['max_stake_percentage'], 0.05)
        else:
            recommended_stake = 0.0
        
        return BettingEdge(
            bet_type=bet_type,
            player_id=player_id,
            edge_percentage=edge,
            confidence_level=confidence_level,
            market_inefficiency=market_inefficiency,
            recommended_stake=recommended_stake
        )
    
    def _get_current_market_line(self, player_id: str, stat_type: str) -> Optional[float]:
        """Get current market line for a player prop."""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT current_line
            FROM line_movements
            WHERE player_id = ? AND stat_type = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        result = conn.execute(query, (player_id, stat_type)).fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def analyze_correlation_opportunities(self, players: List[str]) -> List[Dict[str, Any]]:
        """Analyze correlation opportunities between player props."""
        
        correlations = []
        
        # Same game correlations
        for i, player1 in enumerate(players):
            for player2 in players[i+1:]:
                # Check if same team (positive correlation) or opposing teams (negative correlation)
                team1 = player1.split('_')[-1] if '_' in player1 else 'unk'
                team2 = player2.split('_')[-1] if '_' in player2 else 'unk'
                
                if team1 == team2:
                    correlation_type = "POSITIVE_SAME_TEAM"
                    correlation_strength = 0.3
                else:
                    correlation_type = "NEGATIVE_OPPOSING_TEAMS"
                    correlation_strength = -0.2
                
                correlations.append({
                    'player1': player1,
                    'player2': player2,
                    'correlation_type': correlation_type,
                    'correlation_strength': correlation_strength,
                    'betting_strategy': self._suggest_correlation_strategy(correlation_type, correlation_strength)
                })
        
        return correlations
    
    def _suggest_correlation_strategy(self, correlation_type: str, strength: float) -> str:
        """Suggest betting strategy based on correlation."""
        
        if correlation_type == "POSITIVE_SAME_TEAM" and strength > 0.2:
            return "CONSIDER_SAME_GAME_PARLAY"
        elif correlation_type == "NEGATIVE_OPPOSING_TEAMS" and abs(strength) > 0.15:
            return "CONSIDER_CONTRARIAN_BETS"
        else:
            return "NO_CORRELATION_STRATEGY"
    
    def track_betting_performance(self, bet_results: List[Dict]) -> Dict[str, float]:
        """Track and analyze betting performance metrics."""
        
        if not bet_results:
            return {}
        
        # Calculate performance metrics
        total_bets = len(bet_results)
        wins = sum(1 for bet in bet_results if bet.get('result') == 'WIN')
        losses = sum(1 for bet in bet_results if bet.get('result') == 'LOSS')
        pushes = sum(1 for bet in bet_results if bet.get('result') == 'PUSH')
        
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # ROI calculation
        total_staked = sum(bet.get('stake', 0) for bet in bet_results)
        total_profit = sum(bet.get('profit', 0) for bet in bet_results)
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        
        # Streak analysis
        current_streak = 0
        streak_type = None
        for bet in reversed(bet_results):
            if bet.get('result') == 'WIN':
                if streak_type == 'WIN' or streak_type is None:
                    current_streak += 1
                    streak_type = 'WIN'
                else:
                    break
            elif bet.get('result') == 'LOSS':
                if streak_type == 'LOSS' or streak_type is None:
                    current_streak += 1
                    streak_type = 'LOSS'
                else:
                    break
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'roi': roi,
            'total_profit': total_profit,
            'current_streak': current_streak,
            'streak_type': streak_type,
            'performance_grade': self._grade_performance(win_rate, roi)
        }
    
    def _grade_performance(self, win_rate: float, roi: float) -> str:
        """Grade betting performance."""
        
        if win_rate >= 0.58 and roi >= 8:
            return 'EXCELLENT'
        elif win_rate >= 0.55 and roi >= 5:
            return 'GOOD'
        elif win_rate >= 0.52 and roi >= 2:
            return 'AVERAGE'
        elif win_rate >= 0.48 and roi >= -2:
            return 'BELOW_AVERAGE'
        else:
            return 'POOR'


class PublicBettingAnalyzer:
    """Analyze public betting patterns and fade opportunities."""
    
    def __init__(self):
        self.public_thresholds = {
            'heavy_public': 0.75,      # 75%+ public on one side
            'moderate_public': 0.65,   # 65%+ public on one side
            'contrarian_threshold': 0.8 # Fade public when 80%+ on one side
        }
    
    def analyze_public_betting_patterns(self, betting_data: List[Dict]) -> Dict[str, Any]:
        """Analyze public betting patterns for contrarian opportunities."""
        
        if not betting_data:
            return {}
        
        # Aggregate public betting percentages
        public_heavy_bets = []
        contrarian_opportunities = []
        
        for bet in betting_data:
            public_pct = bet.get('public_percentage', 50)
            
            if public_pct >= self.public_thresholds['heavy_public'] * 100:
                public_heavy_bets.append({
                    'player_id': bet.get('player_id'),
                    'stat_type': bet.get('stat_type'),
                    'public_percentage': public_pct,
                    'side': 'OVER',
                    'contrarian_recommendation': 'FADE_PUBLIC_TAKE_UNDER'
                })
            elif public_pct <= (1 - self.public_thresholds['heavy_public']) * 100:
                public_heavy_bets.append({
                    'player_id': bet.get('player_id'),
                    'stat_type': bet.get('stat_type'),
                    'public_percentage': public_pct,
                    'side': 'UNDER',
                    'contrarian_recommendation': 'FADE_PUBLIC_TAKE_OVER'
                })
            
            # Extreme contrarian opportunities
            if public_pct >= self.public_thresholds['contrarian_threshold'] * 100:
                contrarian_opportunities.append({
                    'player_id': bet.get('player_id'),
                    'stat_type': bet.get('stat_type'),
                    'public_percentage': public_pct,
                    'contrarian_strength': 'STRONG',
                    'recommendation': 'STRONG_FADE_TAKE_UNDER'
                })
        
        return {
            'public_heavy_bets': public_heavy_bets,
            'contrarian_opportunities': contrarian_opportunities,
            'fade_recommendations': len(contrarian_opportunities),
            'public_bias_analysis': self._analyze_public_bias(betting_data)
        }
    
    def _analyze_public_bias(self, betting_data: List[Dict]) -> Dict[str, float]:
        """Analyze overall public betting biases."""
        
        if not betting_data:
            return {}
        
        # Calculate average public percentages by bet type
        over_bets = [bet for bet in betting_data if bet.get('public_percentage', 50) > 50]
        under_bets = [bet for bet in betting_data if bet.get('public_percentage', 50) < 50]
        
        over_bias = len(over_bets) / len(betting_data) if betting_data else 0
        avg_over_percentage = np.mean([bet.get('public_percentage', 50) for bet in over_bets]) if over_bets else 50
        
        return {
            'over_bias_percentage': over_bias,
            'average_over_public_percentage': avg_over_percentage,
            'public_bias_strength': 'STRONG' if over_bias > 0.7 else 'MODERATE' if over_bias > 0.6 else 'WEAK'
        }


# Example usage and integration
def main():
    """Example usage of advanced market analytics."""
    
    print("Advanced Market Analytics Demo")
    print("=" * 40)
    
    # Initialize analyzers
    market_analyzer = AdvancedMarketAnalyzer()
    public_analyzer = PublicBettingAnalyzer()
    
    # Example sharp money detection
    print("\nSharp Money Analysis:")
    sharp_analysis = market_analyzer.detect_sharp_money_movement("pmahomes_qb", "passing_yards")
    if sharp_analysis:
        print(f"Sharp Money Strength: {sharp_analysis.get('strength', 0):.2f}")
        print(f"Recommendation: {sharp_analysis.get('recommendation', 'NO_SIGNAL')}")
    
    # Example value calculation
    print("\nValue Bet Analysis:")
    value_bet = market_analyzer.calculate_market_value("cmccaffrey_rb", "rushing_yards", 95.5, 0.75)
    print(f"Edge: {value_bet.edge_percentage:.3f}")
    print(f"Confidence: {value_bet.confidence_level}")
    print(f"Recommended Stake: {value_bet.recommended_stake:.3f}")
    
    # Example performance tracking
    sample_results = [
        {'result': 'WIN', 'stake': 100, 'profit': 90},
        {'result': 'LOSS', 'stake': 100, 'profit': -100},
        {'result': 'WIN', 'stake': 100, 'profit': 95}
    ]
    
    performance = market_analyzer.track_betting_performance(sample_results)
    print(f"\nPerformance Analysis:")
    print(f"Win Rate: {performance.get('win_rate', 0):.1%}")
    print(f"ROI: {performance.get('roi', 0):.1f}%")
    print(f"Grade: {performance.get('performance_grade', 'N/A')}")
    
    print("\nAdvanced market analytics ready!")


if __name__ == "__main__":
    main()
