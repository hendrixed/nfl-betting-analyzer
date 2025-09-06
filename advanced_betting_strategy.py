#!/usr/bin/env python3
"""
Advanced NFL Betting Strategy System
Includes bankroll management, Kelly Criterion, and game outcome predictions.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from betting_predictor import NFLBettingPredictor
from config_manager import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BetType(Enum):
    PLAYER_PROP = "player_prop"
    GAME_TOTAL = "game_total"
    SPREAD = "spread"
    MONEYLINE = "moneyline"

@dataclass
class BettingRecommendation:
    """Advanced betting recommendation with risk management."""
    player_id: str
    position: str
    bet_type: BetType
    market: str  # e.g., "Over 14.5 fantasy points"
    predicted_value: float
    implied_probability: float
    confidence: float
    kelly_percentage: float
    recommended_units: float
    expected_value: float
    risk_level: str  # LOW, MEDIUM, HIGH

class AdvancedBettingStrategy:
    """Advanced betting strategy with bankroll management and Kelly Criterion."""
    
    def __init__(self, bankroll: float = 1000.0):
        """Initialize with starting bankroll."""
        self.config = get_config()
        self.engine = create_engine(self.config.database.url)
        self.Session = sessionmaker(bind=self.engine)
        self.predictor = NFLBettingPredictor()
        
        # Bankroll management
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.max_bet_percentage = 0.05  # Max 5% of bankroll per bet
        self.min_confidence = 0.65  # Minimum 65% confidence
        self.min_edge = 0.10  # Minimum 10% edge over implied odds
        
        logger.info(f"Initialized advanced betting strategy with ${bankroll:,.2f} bankroll")
    
    def calculate_kelly_criterion(self, win_probability: float, odds: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            win_probability: Probability of winning (0-1)
            odds: Decimal odds (e.g., 1.91 for -110)
        
        Returns:
            Fraction of bankroll to bet (0-1)
        """
        if win_probability <= 0 or odds <= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = win probability, q = lose probability
        b = odds - 1
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at max bet percentage for risk management
        return max(0, min(kelly_fraction, self.max_bet_percentage))
    
    def calculate_implied_probability(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def get_market_odds(self, market_type: str, threshold: float) -> int:
        """
        Simulate market odds for different bet types.
        In a real system, this would fetch from sportsbook APIs.
        """
        # Simulated odds based on common market patterns
        odds_map = {
            'fantasy_points_over_14.5': -110,
            'fantasy_points_over_9.5': -120,
            'fantasy_points_over_19.5': -105,
            'fantasy_points_over_15.5': -115,
            'fantasy_points_over_11.5': -110,
            'fantasy_points_over_7.5': -125,
            'passing_yards_over_274.5': -110,
            'passing_yards_over_224.5': -115,
            'passing_touchdowns_over_2.5': -105,
            'passing_touchdowns_over_1.5': -120,
        }
        
        # Default to -110 if not found
        return odds_map.get(market_type, -110)
    
    def analyze_betting_edge(self, prediction: Dict) -> Optional[BettingRecommendation]:
        """Analyze if a prediction has positive expected value."""
        player_id = prediction['player_id']
        position = prediction['position']
        stat_type = prediction['stat_type']
        predicted_value = prediction['predicted_value']
        confidence = prediction['confidence']
        
        # Skip if confidence too low
        if confidence < self.min_confidence:
            return None
        
        # Determine market and threshold based on prediction
        market_key = None
        threshold = None
        
        if stat_type == 'fantasy_points':
            if position == 'RB' and predicted_value > 14.5:
                market_key = 'fantasy_points_over_14.5'
                threshold = 14.5
            elif position == 'WR' and predicted_value > 11.5:
                market_key = 'fantasy_points_over_11.5'
                threshold = 11.5
            elif position == 'TE' and predicted_value > 9.5:
                market_key = 'fantasy_points_over_9.5'
                threshold = 9.5
            elif position == 'QB' and predicted_value > 19.5:
                market_key = 'fantasy_points_over_19.5'
                threshold = 19.5
        elif stat_type == 'passing_yards' and predicted_value > 274.5:
            market_key = 'passing_yards_over_274.5'
            threshold = 274.5
        elif stat_type == 'passing_touchdowns' and predicted_value > 2.5:
            market_key = 'passing_touchdowns_over_2.5'
            threshold = 2.5
        
        if not market_key:
            return None
        
        # Get market odds and calculate edge
        american_odds = self.get_market_odds(market_key, threshold)
        implied_prob = self.calculate_implied_probability(american_odds)
        decimal_odds = self._american_to_decimal(american_odds)
        
        # Calculate our edge
        edge = confidence - implied_prob
        
        # Skip if edge too small
        if edge < self.min_edge:
            return None
        
        # Calculate Kelly bet size
        kelly_fraction = self.calculate_kelly_criterion(confidence, decimal_odds)
        
        if kelly_fraction <= 0:
            return None
        
        # Calculate expected value
        expected_value = (confidence * (decimal_odds - 1)) - (1 - confidence)
        
        # Determine risk level
        if kelly_fraction > 0.03:
            risk_level = "HIGH"
        elif kelly_fraction > 0.015:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate recommended units (1 unit = 1% of bankroll)
        recommended_units = kelly_fraction * 100
        
        market_description = f"Over {threshold} {stat_type.replace('_', ' ')}"
        
        return BettingRecommendation(
            player_id=player_id,
            position=position,
            bet_type=BetType.PLAYER_PROP,
            market=market_description,
            predicted_value=predicted_value,
            implied_probability=implied_prob,
            confidence=confidence,
            kelly_percentage=kelly_fraction,
            recommended_units=recommended_units,
            expected_value=expected_value,
            risk_level=risk_level
        )
    
    def _american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def get_advanced_recommendations(self) -> List[BettingRecommendation]:
        """Get advanced betting recommendations with bankroll management."""
        # Get basic predictions
        basic_predictions = self.predictor.get_betting_recommendations()
        
        advanced_recs = []
        for prediction in basic_predictions:
            rec = self.analyze_betting_edge(prediction)
            if rec:
                advanced_recs.append(rec)
        
        # Sort by expected value descending
        advanced_recs.sort(key=lambda x: x.expected_value, reverse=True)
        
        return advanced_recs[:10]  # Top 10 value bets
    
    def calculate_portfolio_risk(self, recommendations: List[BettingRecommendation]) -> Dict:
        """Calculate overall portfolio risk metrics."""
        if not recommendations:
            return {}
        
        total_units = sum(rec.recommended_units for rec in recommendations)
        total_bankroll_risk = sum(rec.kelly_percentage for rec in recommendations)
        
        avg_confidence = np.mean([rec.confidence for rec in recommendations])
        avg_expected_value = np.mean([rec.expected_value for rec in recommendations])
        
        risk_distribution = {
            'LOW': len([r for r in recommendations if r.risk_level == 'LOW']),
            'MEDIUM': len([r for r in recommendations if r.risk_level == 'MEDIUM']),
            'HIGH': len([r for r in recommendations if r.risk_level == 'HIGH'])
        }
        
        return {
            'total_recommended_units': total_units,
            'total_bankroll_at_risk': total_bankroll_risk * 100,  # As percentage
            'average_confidence': avg_confidence,
            'average_expected_value': avg_expected_value,
            'risk_distribution': risk_distribution,
            'number_of_bets': len(recommendations)
        }
    
    def display_advanced_strategy(self):
        """Display advanced betting strategy with risk management."""
        print("🎯 ADVANCED NFL BETTING STRATEGY")
        print("=" * 60)
        print(f"💰 Current Bankroll: ${self.current_bankroll:,.2f}")
        print(f"📊 Max Bet Size: {self.max_bet_percentage*100:.1f}% of bankroll")
        print(f"🎲 Min Confidence: {self.min_confidence*100:.0f}%")
        print(f"📈 Min Edge Required: {self.min_edge*100:.0f}%")
        print()
        
        recommendations = self.get_advanced_recommendations()
        
        if not recommendations:
            print("❌ No value bets found with current criteria.")
            return
        
        # Display portfolio metrics
        portfolio = self.calculate_portfolio_risk(recommendations)
        print("📊 PORTFOLIO ANALYSIS:")
        print("-" * 40)
        print(f"Total Recommended Units: {portfolio['total_recommended_units']:.1f}")
        print(f"Total Bankroll at Risk: {portfolio['total_bankroll_at_risk']:.1f}%")
        print(f"Average Confidence: {portfolio['average_confidence']:.1%}")
        print(f"Average Expected Value: {portfolio['average_expected_value']:.2%}")
        print(f"Risk Distribution: {portfolio['risk_distribution']}")
        print()
        
        # Display individual recommendations
        print("🎯 VALUE BET RECOMMENDATIONS:")
        print("-" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            risk_emoji = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}
            position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
            
            bet_amount = rec.kelly_percentage * self.current_bankroll
            
            print(f"{i}. {rec.player_id} ({rec.position}) {position_emoji.get(rec.position, '⚡')}")
            print(f"   Market: {rec.market}")
            print(f"   Predicted: {rec.predicted_value:.1f}")
            print(f"   Confidence: {rec.confidence:.1%} | Implied: {rec.implied_probability:.1%}")
            print(f"   Edge: {(rec.confidence - rec.implied_probability):.1%}")
            print(f"   Expected Value: {rec.expected_value:.2%}")
            print(f"   Kelly %: {rec.kelly_percentage:.2%}")
            print(f"   Recommended: {rec.recommended_units:.1f} units (${bet_amount:.2f})")
            print(f"   Risk Level: {rec.risk_level} {risk_emoji.get(rec.risk_level, '⚪')}")
            print()
        
        print("⚠️  RISK DISCLAIMER:")
        print("   • These are mathematical models, not guarantees")
        print("   • Never bet more than you can afford to lose")
        print("   • Consider this entertainment, not investment advice")
        print("   • Past performance doesn't predict future results")

def main():
    """Main function for advanced betting strategy."""
    # Initialize with $1000 bankroll (adjust as needed)
    strategy = AdvancedBettingStrategy(bankroll=1000.0)
    strategy.display_advanced_strategy()

if __name__ == "__main__":
    main()
