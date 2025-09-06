#!/usr/bin/env python3
"""
NFL Betting Analyzer - Interactive Main Interface
One-stop interface for all NFL betting analysis and predictions
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the most advanced components
try:
    from ultimate_enhanced_predictor import UltimateEnhancedPredictor
    from social_sentiment_analyzer import SocialSentimentAnalyzer, NewsImpactAnalyzer
    from config_manager import get_config
    from run_nfl_system import cli as system_cli
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Some advanced features may not be available.")

logger = logging.getLogger(__name__)

class NFLInteractiveInterface:
    """Interactive interface for NFL betting analysis."""
    
    def __init__(self):
        try:
            self.predictor = UltimateEnhancedPredictor()
            self.sentiment_analyzer = SocialSentimentAnalyzer()
            self.news_analyzer = NewsImpactAnalyzer()
            self.config = get_config()
            print("✅ All advanced components loaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Some components failed to load: {e}")
            self.predictor = None
            self.sentiment_analyzer = None
            self.news_analyzer = None
        
    def show_main_menu(self):
        """Display the main interactive menu."""
        print("\n" + "="*70)
        print("🏈 NFL BETTING ANALYZER - INTERACTIVE INTERFACE")
        print("="*70)
        print()
        print("📊 PLAYER ANALYSIS:")
        print("   1. Get Ultimate Player Predictions (all stats)")
        print("   2. Compare Multiple Players")
        print("   3. Player Sentiment Analysis")
        print()
        print("🎯 GAME ANALYSIS:")
        print("   4. Game Predictions & Betting Lines")
        print("   5. Team Matchup Analysis")
        print("   6. Weather & Situation Impact")
        print()
        print("💰 BETTING RECOMMENDATIONS:")
        print("   7. Daily Top Betting Opportunities")
        print("   8. Prop Bet Recommendations")
        print("   9. Value Bet Scanner")
        print("   10. Portfolio Optimizer")
        print()
        print("📈 ADVANCED ANALYTICS:")
        print("   11. Market Intelligence Report")
        print("   12. Social Sentiment Deep Dive")
        print("   13. Injury Impact Assessment")
        print("   14. Line Movement Tracker")
        print()
        print("⚙️  SYSTEM MANAGEMENT:")
        print("   15. System Status & Health Check")
        print("   16. Update Data & Retrain Models")
        print("   17. Configuration Settings")
        print("   18. Run Full System Demo")
        print()
        print("❌ EXIT:")
        print("   0. Exit Application")
        print()
        
    async def run_interactive_session(self):
        """Run the interactive session."""
        print("🚀 Starting NFL Betting Analyzer Interactive Interface...")
        
        if not self.predictor:
            print("❌ Critical components not loaded. Please check your installation.")
            return
        
        while True:
            self.show_main_menu()
            
            try:
                choice = input("🔸 Enter your choice (0-18): ").strip()
                
                if choice == "0":
                    print("\n👋 Thank you for using NFL Betting Analyzer!")
                    print("Remember to gamble responsibly! 🎯")
                    break
                elif choice == "1":
                    await self.handle_player_predictions()
                elif choice == "2":
                    await self.handle_player_comparison()
                elif choice == "3":
                    await self.handle_sentiment_analysis()
                elif choice == "4":
                    await self.handle_game_predictions()
                elif choice == "5":
                    await self.handle_team_matchup()
                elif choice == "6":
                    await self.handle_weather_impact()
                elif choice == "7":
                    await self.handle_daily_opportunities()
                elif choice == "8":
                    await self.handle_prop_recommendations()
                elif choice == "9":
                    await self.handle_value_scanner()
                elif choice == "10":
                    await self.handle_portfolio_optimizer()
                elif choice == "11":
                    await self.handle_market_intelligence()
                elif choice == "12":
                    await self.handle_sentiment_deep_dive()
                elif choice == "13":
                    await self.handle_injury_assessment()
                elif choice == "14":
                    await self.handle_line_movement()
                elif choice == "15":
                    await self.handle_system_status()
                elif choice == "16":
                    await self.handle_data_update()
                elif choice == "17":
                    await self.handle_configuration()
                elif choice == "18":
                    await self.handle_demo()
                else:
                    print("❌ Invalid choice. Please enter a number between 0-18.")
                    
                input("\n⏸️  Press Enter to continue...")
                print("\n" * 2)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("\n⏸️  Press Enter to continue...")
    
    async def handle_player_predictions(self):
        """Handle ultimate player prediction requests."""
        print("\n🔍 ULTIMATE PLAYER PREDICTIONS")
        print("-" * 40)
        
        player_id = input("Enter player ID (e.g., 'pmahomes_qb'): ").strip()
        if not player_id:
            print("❌ No player ID provided.")
            return
            
        opponent = input("Enter opponent team (optional): ").strip() or None
        
        print(f"\n📊 Generating ultimate analysis for {player_id}...")
        
        try:
            # Use the ultimate predictor's display method
            self.predictor.display_ultimate_analysis(player_id, opponent)
        except Exception as e:
            print(f"❌ Error generating prediction: {e}")
    
    async def handle_player_comparison(self):
        """Handle multiple player comparison."""
        print("\n🔍 PLAYER COMPARISON")
        print("-" * 30)
        
        print("Enter player IDs separated by commas:")
        print("Example: pmahomes_qb,jallen_qb,lburrow_qb")
        players_input = input("Players: ").strip()
        
        if not players_input:
            print("❌ No players provided.")
            return
        
        player_ids = [p.strip() for p in players_input.split(",")]
        
        print(f"\n📊 Comparing {len(player_ids)} players...")
        
        try:
            comparison_df = self.predictor.compare_multiple_players(player_ids)
            if not comparison_df.empty:
                print("\n📈 COMPARISON RESULTS:")
                print("=" * 80)
                print(comparison_df.to_string(index=False))
            else:
                print("❌ No comparison data available.")
        except Exception as e:
            print(f"❌ Error in comparison: {e}")
    
    async def handle_sentiment_analysis(self):
        """Handle sentiment analysis for a player."""
        print("\n🔍 PLAYER SENTIMENT ANALYSIS")
        print("-" * 35)
        
        player_id = input("Enter player ID: ").strip()
        if not player_id:
            print("❌ No player ID provided.")
            return
        
        try:
            print(f"\n📊 Analyzing sentiment for {player_id}...")
            
            sentiment_data = self.sentiment_analyzer.analyze_player_sentiment(player_id)
            multiplier = self.sentiment_analyzer.get_sentiment_multiplier(player_id)
            injury_analysis = self.sentiment_analyzer.analyze_injury_sentiment(player_id)
            
            print(f"\n📈 SENTIMENT ANALYSIS RESULTS:")
            print("=" * 50)
            print(f"  Sentiment Score: {sentiment_data.sentiment_score:.3f}")
            print(f"  Mention Volume: {sentiment_data.mention_volume:,}")
            print(f"  Positive/Negative/Neutral: {sentiment_data.positive_mentions}/{sentiment_data.negative_mentions}/{sentiment_data.neutral_mentions}")
            print(f"  Trending Topics: {', '.join(sentiment_data.trending_topics)}")
            print(f"  Sentiment Multiplier: {multiplier:.3f}")
            print(f"  Injury Concern Level: {injury_analysis['injury_concern_level']}")
            print(f"  News Impact Score: {sentiment_data.news_impact_score:.3f}")
            
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
    
    async def handle_game_predictions(self):
        """Handle game predictions."""
        print("\n🎯 GAME PREDICTIONS")
        print("-" * 25)
        print("🚧 Feature coming soon! This will provide:")
        print("   • Full game outcome predictions")
        print("   • Betting line recommendations")
        print("   • Over/under analysis")
        print("   • Spread predictions")
    
    async def handle_team_matchup(self):
        """Handle team matchup analysis."""
        print("\n🏈 TEAM MATCHUP ANALYSIS")
        print("-" * 30)
        print("🚧 Feature coming soon! This will analyze:")
        print("   • Team vs team historical performance")
        print("   • Offensive vs defensive matchups")
        print("   • Key player matchups")
        print("   • Coaching tendencies")
    
    async def handle_weather_impact(self):
        """Handle weather impact analysis."""
        print("\n🌤️  WEATHER & SITUATION IMPACT")
        print("-" * 35)
        print("🚧 Feature coming soon! This will include:")
        print("   • Weather impact on player performance")
        print("   • Dome vs outdoor game analysis")
        print("   • Temperature and wind effects")
        print("   • Prime time game adjustments")
    
    async def handle_daily_opportunities(self):
        """Handle daily betting opportunities."""
        print("\n💰 DAILY TOP BETTING OPPORTUNITIES")
        print("-" * 40)
        print("🚧 Feature coming soon! This will provide:")
        print("   • Top value bets of the day")
        print("   • Highest confidence plays")
        print("   • Contrarian opportunities")
        print("   • Risk-adjusted recommendations")
    
    async def handle_prop_recommendations(self):
        """Handle prop bet recommendations."""
        print("\n🎯 PROP BET RECOMMENDATIONS")
        print("-" * 35)
        print("🚧 Feature coming soon! This will offer:")
        print("   • Player prop recommendations")
        print("   • Game prop analysis")
        print("   • Same-game parlay suggestions")
        print("   • Value prop identification")
    
    async def handle_value_scanner(self):
        """Handle value bet scanner."""
        print("\n🔍 VALUE BET SCANNER")
        print("-" * 25)
        print("🚧 Feature coming soon! This will scan for:")
        print("   • Market inefficiencies")
        print("   • Line shopping opportunities")
        print("   • Arbitrage possibilities")
        print("   • Expected value calculations")
    
    async def handle_portfolio_optimizer(self):
        """Handle portfolio optimization."""
        print("\n📊 PORTFOLIO OPTIMIZER")
        print("-" * 28)
        print("🚧 Feature coming soon! This will optimize:")
        print("   • Bankroll allocation")
        print("   • Risk management")
        print("   • Kelly criterion betting")
        print("   • Diversification strategies")
    
    async def handle_market_intelligence(self):
        """Handle market intelligence report."""
        print("\n📈 MARKET INTELLIGENCE REPORT")
        print("-" * 38)
        print("🚧 Feature coming soon! This will track:")
        print("   • Sharp money movement")
        print("   • Public betting percentages")
        print("   • Line movement patterns")
        print("   • Reverse line movement")
    
    async def handle_sentiment_deep_dive(self):
        """Handle deep sentiment analysis."""
        print("\n🔍 SOCIAL SENTIMENT DEEP DIVE")
        print("-" * 38)
        print("🚧 Feature coming soon! This will analyze:")
        print("   • Twitter sentiment trends")
        print("   • News impact analysis")
        print("   • Reddit discussion sentiment")
        print("   • Contrarian betting opportunities")
    
    async def handle_injury_assessment(self):
        """Handle injury impact assessment."""
        print("\n🏥 INJURY IMPACT ASSESSMENT")
        print("-" * 35)
        print("🚧 Feature coming soon! This will assess:")
        print("   • Real-time injury reports")
        print("   • Historical injury impact")
        print("   • Recovery timelines")
        print("   • Replacement player analysis")
    
    async def handle_line_movement(self):
        """Handle line movement tracking."""
        print("\n📊 LINE MOVEMENT TRACKER")
        print("-" * 30)
        print("🚧 Feature coming soon! This will track:")
        print("   • Real-time line movements")
        print("   • Historical line patterns")
        print("   • Steam moves identification")
        print("   • Reverse line movement alerts")
    
    async def handle_system_status(self):
        """Handle system status check."""
        print("\n⚙️  SYSTEM STATUS & HEALTH CHECK")
        print("-" * 40)
        
        try:
            # Import and run system status
            import subprocess
            result = subprocess.run([
                sys.executable, "run_nfl_system.py", "status"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ System status check failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error checking system status: {e}")
    
    async def handle_data_update(self):
        """Handle data update and model retraining."""
        print("\n🔄 UPDATE DATA & RETRAIN MODELS")
        print("-" * 40)
        
        print("Available update options:")
        print("  1. Download latest NFL data")
        print("  2. Retrain ML models")
        print("  3. Update configuration")
        print("  4. Full system update")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        try:
            import subprocess
            
            if choice == "1":
                print("📥 Downloading latest NFL data...")
                result = subprocess.run([
                    sys.executable, "run_nfl_system.py", "download-data"
                ], cwd=Path(__file__).parent)
            elif choice == "2":
                print("🤖 Retraining ML models...")
                result = subprocess.run([
                    sys.executable, "run_nfl_system.py", "train"
                ], cwd=Path(__file__).parent)
            elif choice == "3":
                print("⚙️  Configuration update not yet implemented.")
            elif choice == "4":
                print("🔄 Running full system update...")
                result = subprocess.run([
                    sys.executable, "run_nfl_system.py", "setup"
                ], cwd=Path(__file__).parent)
            else:
                print("❌ Invalid choice.")
                
        except Exception as e:
            print(f"❌ Error during update: {e}")
    
    async def handle_configuration(self):
        """Handle configuration settings."""
        print("\n⚙️  CONFIGURATION SETTINGS")
        print("-" * 35)
        
        try:
            print("Current Configuration:")
            print(f"  Version: {self.config.version}")
            print(f"  Environment: {self.config.environment}")
            print(f"  Database Type: {self.config.database.type}")
            print(f"  API Host: {self.config.api.host}:{self.config.api.port}")
            print(f"  Model Directory: {self.config.models.directory}")
            print(f"  Data Directory: {self.config.data.directory}")
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
    
    async def handle_demo(self):
        """Handle full system demo."""
        print("\n🎬 FULL SYSTEM DEMO")
        print("-" * 25)
        
        try:
            import subprocess
            print("🚀 Starting comprehensive system demo...")
            
            result = subprocess.run([
                sys.executable, "demo.py"
            ], cwd=Path(__file__).parent)
            
            if result.returncode != 0:
                print("⚠️  Demo completed with warnings. Check output above.")
            else:
                print("✅ Demo completed successfully!")
                
        except Exception as e:
            print(f"❌ Error running demo: {e}")


def main():
    """Main entry point for interactive interface."""
    try:
        interface = NFLInteractiveInterface()
        asyncio.run(interface.run_interactive_session())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()
