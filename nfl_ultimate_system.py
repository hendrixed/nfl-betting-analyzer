#!/usr/bin/env python3
"""
NFL Ultimate System - Consolidated Advanced Analytics
Combines all advanced features into one comprehensive system
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all advanced components
try:
    from ultimate_enhanced_predictor import UltimateEnhancedPredictor, UltimatePrediction
    from social_sentiment_analyzer import SocialSentimentAnalyzer, NewsImpactAnalyzer
    from advanced_market_analytics import AdvancedMarketAnalyzer, PublicBettingAnalyzer
    from config_manager import get_config
except ImportError as e:
    print(f"⚠️  Import warning: {e}")

logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveAnalysis:
    """Complete analysis combining all system components."""
    player_id: str
    position: str
    ultimate_prediction: UltimatePrediction
    sentiment_data: Dict[str, Any]
    market_intelligence: Dict[str, Any]
    final_recommendation: Dict[str, Any]
    confidence_level: str
    risk_assessment: str

class NFLUltimateSystem:
    """The ultimate NFL betting analysis system."""
    
    def __init__(self, db_path: str = "data/nfl_predictions.db"):
        """Initialize all advanced components."""
        self.db_path = db_path
        
        try:
            # Initialize all advanced components
            self.predictor = UltimateEnhancedPredictor(db_path)
            self.sentiment = SocialSentimentAnalyzer()
            self.news_analyzer = NewsImpactAnalyzer()
            self.market = AdvancedMarketAnalyzer()
            self.public_betting = PublicBettingAnalyzer()
            self.config = get_config()
            
            logger.info("✅ All ultimate system components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
        
    def get_complete_player_analysis(self, player_id: str, opponent: str = None) -> ComprehensiveAnalysis:
        """Get the most comprehensive player analysis possible."""
        
        logger.info(f"🔍 Running complete analysis for {player_id}")
        
        try:
            # 1. Ultimate Prediction (ML + Situational + Market)
            ultimate_prediction = self.predictor.generate_ultimate_prediction(player_id, opponent)
            
            # 2. Sentiment Analysis
            sentiment_data = self._get_comprehensive_sentiment(player_id)
            
            # 3. Market Intelligence
            market_intelligence = self._get_market_intelligence(player_id)
            
            # 4. Generate Final Recommendation
            final_recommendation = self._generate_final_recommendation(
                ultimate_prediction, sentiment_data, market_intelligence
            )
            
            # 5. Assess Overall Confidence and Risk
            confidence_level = self._assess_overall_confidence(
                ultimate_prediction, sentiment_data, market_intelligence
            )
            risk_assessment = self._assess_comprehensive_risk(
                ultimate_prediction, sentiment_data, market_intelligence
            )
            
            return ComprehensiveAnalysis(
                player_id=player_id,
                position=ultimate_prediction.position,
                ultimate_prediction=ultimate_prediction,
                sentiment_data=sentiment_data,
                market_intelligence=market_intelligence,
                final_recommendation=final_recommendation,
                confidence_level=confidence_level,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"❌ Complete analysis failed for {player_id}: {e}")
            raise
    
    def _get_comprehensive_sentiment(self, player_id: str) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis."""
        try:
            sentiment_data = self.sentiment.analyze_player_sentiment(player_id)
            sentiment_multiplier = self.sentiment.get_sentiment_multiplier(player_id)
            injury_sentiment = self.sentiment.analyze_injury_sentiment(player_id)
            
            # Mock news headlines for demonstration
            recent_headlines = [
                f"{player_id.split('_')[0].title()} looks sharp in practice",
                f"Coach optimistic about {player_id.split('_')[0].title()}'s performance"
            ]
            news_multiplier = self.news_analyzer.get_news_multiplier(player_id, recent_headlines)
            
            return {
                'sentiment_score': sentiment_data.sentiment_score,
                'mention_volume': sentiment_data.mention_volume,
                'trending_topics': sentiment_data.trending_topics,
                'sentiment_multiplier': sentiment_multiplier,
                'news_multiplier': news_multiplier,
                'injury_analysis': injury_sentiment,
                'overall_sentiment_impact': (sentiment_multiplier + news_multiplier) / 2
            }
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {'sentiment_score': 0, 'overall_sentiment_impact': 1.0}
    
    def _get_market_intelligence(self, player_id: str) -> Dict[str, Any]:
        """Get comprehensive market intelligence."""
        try:
            # Mock market data for demonstration
            market_data = {
                'current_line': 18.5,
                'opening_line': 19.0,
                'line_movement': -0.5,
                'sharp_money_indicator': 'NEUTRAL',
                'public_betting_percentage': 65.0,
                'handle_percentage': 58.0,
                'market_efficiency': 0.85,
                'value_opportunity': 'MODERATE'
            }
            
            return market_data
            
        except Exception as e:
            logger.warning(f"Market intelligence failed: {e}")
            return {'market_efficiency': 0.5, 'value_opportunity': 'UNKNOWN'}
    
    def _generate_final_recommendation(self, prediction: UltimatePrediction, 
                                     sentiment: Dict, market: Dict) -> Dict[str, Any]:
        """Generate final betting recommendation combining all factors."""
        
        # Base recommendation from ultimate predictor
        base_recs = prediction.betting_recommendations
        
        # Adjust based on sentiment
        sentiment_adjustment = sentiment.get('overall_sentiment_impact', 1.0)
        
        # Adjust based on market intelligence
        market_adjustment = market.get('market_efficiency', 0.5)
        
        # Calculate overall recommendation strength
        fp_prediction = prediction.final_prediction.get('fantasy_points_ppr', 0)
        confidence = prediction.confidence_score
        market_edge = prediction.market_edge
        
        # Generate recommendation score (0-100)
        rec_score = (
            confidence * 40 +  # 40% weight on confidence
            abs(market_edge) * 100 * 30 +  # 30% weight on market edge
            sentiment_adjustment * 20 +  # 20% weight on sentiment
            market_adjustment * 10  # 10% weight on market efficiency
        )
        
        # Determine recommendation level
        if rec_score >= 80:
            rec_level = "STRONG_BUY"
            rec_description = "High confidence play with multiple positive factors"
        elif rec_score >= 65:
            rec_level = "BUY"
            rec_description = "Good play with solid fundamentals"
        elif rec_score >= 50:
            rec_level = "HOLD"
            rec_description = "Neutral play, proceed with caution"
        elif rec_score >= 35:
            rec_level = "WEAK_SELL"
            rec_description = "Below average play, consider alternatives"
        else:
            rec_level = "STRONG_SELL"
            rec_description = "Avoid this play, multiple negative factors"
        
        return {
            'recommendation_level': rec_level,
            'recommendation_score': rec_score,
            'description': rec_description,
            'base_recommendations': base_recs,
            'key_factors': [
                f"Confidence: {confidence:.1%}",
                f"Market Edge: {market_edge:+.1%}",
                f"Sentiment Impact: {sentiment_adjustment:.3f}",
                f"Predicted FP: {fp_prediction:.1f}"
            ],
            'betting_strategy': self._generate_betting_strategy(rec_level, fp_prediction, confidence)
        }
    
    def _generate_betting_strategy(self, rec_level: str, fp_prediction: float, confidence: float) -> List[str]:
        """Generate specific betting strategy recommendations."""
        strategies = []
        
        if rec_level in ["STRONG_BUY", "BUY"]:
            strategies.append("Consider over bets on player props")
            strategies.append("Strong DFS play consideration")
            if confidence > 0.8:
                strategies.append("Suitable for larger bet sizing")
        elif rec_level == "HOLD":
            strategies.append("Neutral DFS play")
            strategies.append("Wait for better line movement")
        else:
            strategies.append("Consider under bets or fade in DFS")
            strategies.append("Look for alternative players")
        
        return strategies
    
    def _assess_overall_confidence(self, prediction: UltimatePrediction, 
                                 sentiment: Dict, market: Dict) -> str:
        """Assess overall confidence level."""
        
        base_confidence = prediction.confidence_score
        sentiment_boost = sentiment.get('overall_sentiment_impact', 1.0) - 1.0
        market_boost = market.get('market_efficiency', 0.5)
        
        overall_confidence = base_confidence + sentiment_boost * 0.1 + market_boost * 0.1
        
        if overall_confidence >= 0.85:
            return "VERY_HIGH"
        elif overall_confidence >= 0.75:
            return "HIGH"
        elif overall_confidence >= 0.65:
            return "MEDIUM"
        elif overall_confidence >= 0.50:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _assess_comprehensive_risk(self, prediction: UltimatePrediction, 
                                 sentiment: Dict, market: Dict) -> str:
        """Assess comprehensive risk level."""
        
        base_risk = prediction.risk_assessment
        sentiment_volatility = abs(sentiment.get('sentiment_score', 0))
        market_volatility = abs(market.get('line_movement', 0)) / 5.0  # Normalize
        
        # Convert base risk to numeric
        risk_mapping = {"LOW_RISK": 0.2, "MEDIUM_RISK": 0.5, "HIGH_RISK": 0.8}
        base_risk_score = risk_mapping.get(base_risk, 0.5)
        
        # Adjust risk based on volatility
        total_risk = base_risk_score + sentiment_volatility * 0.1 + market_volatility * 0.1
        
        if total_risk <= 0.3:
            return "LOW_RISK"
        elif total_risk <= 0.6:
            return "MEDIUM_RISK"
        else:
            return "HIGH_RISK"
    
    def display_comprehensive_analysis(self, player_id: str, opponent: str = None):
        """Display comprehensive analysis in formatted output."""
        
        try:
            analysis = self.get_complete_player_analysis(player_id, opponent)
            
            print(f"\n{'='*80}")
            print(f"🏈 COMPREHENSIVE NFL ANALYSIS: {player_id.upper()}")
            print(f"{'='*80}")
            
            # Player Info
            print(f"\n📋 PLAYER INFO:")
            print(f"   Position: {analysis.position}")
            print(f"   Overall Confidence: {analysis.confidence_level}")
            print(f"   Risk Assessment: {analysis.risk_assessment}")
            
            # Ultimate Prediction Summary
            pred = analysis.ultimate_prediction
            print(f"\n🔮 PREDICTION SUMMARY:")
            print(f"   Fantasy Points (PPR): {pred.final_prediction.get('fantasy_points_ppr', 0):.1f}")
            print(f"   Confidence Score: {pred.confidence_score:.1%}")
            print(f"   Market Edge: {pred.market_edge:+.1%}")
            print(f"   Value Rating: {pred.value_rating}")
            
            # Sentiment Analysis
            sentiment = analysis.sentiment_data
            print(f"\n💭 SENTIMENT ANALYSIS:")
            print(f"   Sentiment Score: {sentiment.get('sentiment_score', 0):.3f}")
            print(f"   Mention Volume: {sentiment.get('mention_volume', 0):,}")
            print(f"   Trending Topics: {', '.join(sentiment.get('trending_topics', []))}")
            print(f"   Overall Impact: {sentiment.get('overall_sentiment_impact', 1.0):.3f}")
            
            # Market Intelligence
            market = analysis.market_intelligence
            print(f"\n📊 MARKET INTELLIGENCE:")
            print(f"   Current Line: {market.get('current_line', 0):.1f}")
            print(f"   Line Movement: {market.get('line_movement', 0):+.1f}")
            print(f"   Public Betting: {market.get('public_betting_percentage', 0):.1f}%")
            print(f"   Sharp Money: {market.get('sharp_money_indicator', 'UNKNOWN')}")
            
            # Final Recommendation
            rec = analysis.final_recommendation
            print(f"\n💰 FINAL RECOMMENDATION:")
            print(f"   Level: {rec['recommendation_level']}")
            print(f"   Score: {rec['recommendation_score']:.1f}/100")
            print(f"   Description: {rec['description']}")
            
            print(f"\n🎯 KEY FACTORS:")
            for factor in rec['key_factors']:
                print(f"   • {factor}")
            
            print(f"\n📈 BETTING STRATEGY:")
            for strategy in rec['betting_strategy']:
                print(f"   • {strategy}")
            
            print(f"\n⚠️  DISCLAIMER:")
            print(f"   This analysis is for entertainment purposes only.")
            print(f"   Always gamble responsibly and within your means.")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"❌ Error displaying analysis: {e}")
    
    def batch_analyze_players(self, player_ids: List[str]) -> pd.DataFrame:
        """Analyze multiple players and return comparison DataFrame."""
        
        results = []
        
        for player_id in player_ids:
            try:
                analysis = self.get_complete_player_analysis(player_id)
                
                results.append({
                    'Player': player_id,
                    'Position': analysis.position,
                    'FP_Prediction': analysis.ultimate_prediction.final_prediction.get('fantasy_points_ppr', 0),
                    'Confidence': analysis.ultimate_prediction.confidence_score,
                    'Market_Edge': analysis.ultimate_prediction.market_edge,
                    'Sentiment_Score': analysis.sentiment_data.get('sentiment_score', 0),
                    'Recommendation': analysis.final_recommendation['recommendation_level'],
                    'Rec_Score': analysis.final_recommendation['recommendation_score'],
                    'Risk_Level': analysis.risk_assessment
                })
                
            except Exception as e:
                logger.error(f"Failed to analyze {player_id}: {e}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('Rec_Score', ascending=False)
        
        return df


def main():
    """Example usage of the ultimate system."""
    
    print("🚀 NFL ULTIMATE SYSTEM - COMPREHENSIVE ANALYTICS")
    print("=" * 60)
    
    try:
        # Initialize the ultimate system
        system = NFLUltimateSystem()
        
        # Example single player comprehensive analysis
        print("\n🔍 COMPREHENSIVE PLAYER ANALYSIS:")
        system.display_comprehensive_analysis("pmahomes_qb", "den")
        
        print("\n" + "="*80)
        
        # Example batch analysis
        print("\n📊 BATCH PLAYER ANALYSIS:")
        sample_players = ["pmahomes_qb", "jallen_qb", "cmccaffrey_rb", "jchase_wr"]
        
        comparison_df = system.batch_analyze_players(sample_players)
        if not comparison_df.empty:
            print(comparison_df.to_string(index=False))
        
        print("\n✅ Ultimate system demonstration complete!")
        
    except Exception as e:
        print(f"❌ Ultimate system failed: {e}")


if __name__ == "__main__":
    main()
