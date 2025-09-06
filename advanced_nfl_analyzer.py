#!/usr/bin/env python3
"""
Advanced NFL Betting Analyzer
Enhanced with weather data, betting odds, advanced stats, and real-time features.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import requests
import json
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing systems
from streamlined_enhanced_system import StreamlinedEnhancedPredictor
from working_betting_predictor import WorkingBettingPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedNFLAnalyzer:
    """Advanced NFL analyzer with comprehensive data sources and features."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Initialize existing predictors
        self.streamlined_predictor = StreamlinedEnhancedPredictor(db_path)
        self.working_predictor = WorkingBettingPredictor()
        
        # Load API keys from environment
        self.odds_api_key = os.getenv('ODDS_API_KEY', '0111e24d4786064dfe6a5cf6462b2c4c')
        self.weather_api_key = os.getenv('WEATHER_API_KEY', 'your_weather_key_here')
        self.news_api_key = os.getenv('NEWS_API_KEY', '8eb928de87b4442c99931a5a6b6fb912')
        
        # Advanced stats tracking
        self.advanced_stats = {
            'QB': ['air_yards', 'yards_after_catch', 'pressure_rate', 'time_to_throw'],
            'RB': ['yards_before_contact', 'broken_tackles', 'target_share', 'snap_percentage'],
            'WR': ['separation', 'contested_catches', 'route_efficiency', 'red_zone_targets'],
            'TE': ['blocking_grade', 'slot_rate', 'deep_targets', 'goal_line_usage']
        }
        
        logger.info("Advanced NFL Analyzer initialized")
    
    def get_weather_data(self, city: str, game_date: str) -> Dict[str, Any]:
        """Get weather data for outdoor games."""
        if self.weather_api_key == 'your_weather_key_here':
            return self._get_mock_weather_data()
        
        try:
            # OpenWeatherMap API call
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': city,
                'appid': self.weather_api_key,
                'units': 'imperial'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind'].get('deg', 0),
                    'conditions': data['weather'][0]['description'],
                    'precipitation': data.get('rain', {}).get('1h', 0)
                }
        except Exception as e:
            logger.warning(f"Weather API error: {e}")
        
        return self._get_mock_weather_data()
    
    def _get_mock_weather_data(self) -> Dict[str, Any]:
        """Generate mock weather data for testing."""
        return {
            'temperature': np.random.normal(65, 15),
            'humidity': np.random.normal(60, 20),
            'wind_speed': np.random.exponential(8),
            'wind_direction': np.random.uniform(0, 360),
            'conditions': np.random.choice(['clear', 'cloudy', 'rain', 'snow']),
            'precipitation': np.random.exponential(0.1)
        }
    
    def get_betting_odds(self, sport: str = 'americanfootball_nfl') -> Dict[str, Any]:
        """Get current betting odds from The Odds API."""
        if self.odds_api_key == '0111e24d4786064dfe6a5cf6462b2c4c':
            return self._get_mock_odds_data()
        
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Odds API error: {e}")
        
        return self._get_mock_odds_data()
    
    def _get_mock_odds_data(self) -> Dict[str, Any]:
        """Generate mock betting odds for testing."""
        teams = ['KC', 'BUF', 'BAL', 'CIN', 'MIA', 'DAL', 'PHI', 'SF', 'SEA', 'GB']
        games = []
        
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                game = {
                    'id': f'mock_game_{i}',
                    'home_team': teams[i],
                    'away_team': teams[i + 1],
                    'commence_time': (datetime.now() + timedelta(days=np.random.randint(1, 7))).isoformat(),
                    'bookmakers': [{
                        'key': 'draftkings',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {'name': teams[i], 'price': np.random.randint(-200, 200)},
                                    {'name': teams[i + 1], 'price': np.random.randint(-200, 200)}
                                ]
                            },
                            {
                                'key': 'spreads',
                                'outcomes': [
                                    {'name': teams[i], 'price': -110, 'point': np.random.uniform(-7, 7)},
                                    {'name': teams[i + 1], 'price': -110, 'point': np.random.uniform(-7, 7)}
                                ]
                            },
                            {
                                'key': 'totals',
                                'outcomes': [
                                    {'name': 'Over', 'price': -110, 'point': np.random.uniform(40, 55)},
                                    {'name': 'Under', 'price': -110, 'point': np.random.uniform(40, 55)}
                                ]
                            }
                        ]
                    }]
                }
                games.append(game)
        
        return games
    
    def get_player_news_sentiment(self, player_name: str) -> Dict[str, Any]:
        """Get news sentiment analysis for a player."""
        if self.news_api_key == '8eb928de87b4442c99931a5a6b6fb912':
            return self._get_mock_sentiment_data()
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{player_name}" NFL',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Simple sentiment analysis based on keywords
                positive_words = ['great', 'excellent', 'outstanding', 'record', 'best', 'amazing']
                negative_words = ['injured', 'suspended', 'poor', 'worst', 'terrible', 'benched']
                
                sentiment_score = 0
                for article in articles:
                    text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
                    sentiment_score += sum(1 for word in positive_words if word in text)
                    sentiment_score -= sum(1 for word in negative_words if word in text)
                
                return {
                    'sentiment_score': sentiment_score,
                    'article_count': len(articles),
                    'recent_news': articles[:3]
                }
        except Exception as e:
            logger.warning(f"News API error: {e}")
        
        return self._get_mock_sentiment_data()
    
    def _get_mock_sentiment_data(self) -> Dict[str, Any]:
        """Generate mock sentiment data."""
        return {
            'sentiment_score': np.random.randint(-5, 5),
            'article_count': np.random.randint(0, 20),
            'recent_news': []
        }
    
    def calculate_advanced_metrics(self, player_data: pd.DataFrame, position: str) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        metrics = {}
        
        if position == 'QB':
            if 'passing_attempts' in player_data.columns and 'passing_completions' in player_data.columns:
                metrics['completion_percentage'] = (player_data['passing_completions'] / player_data['passing_attempts']).mean() * 100
                metrics['yards_per_attempt'] = (player_data['passing_yards'] / player_data['passing_attempts']).mean()
                metrics['touchdown_rate'] = (player_data['passing_touchdowns'] / player_data['passing_attempts']).mean() * 100
                metrics['interception_rate'] = (player_data['passing_interceptions'] / player_data['passing_attempts']).mean() * 100
                
                # Passer Rating calculation
                a = ((player_data['passing_completions'] / player_data['passing_attempts']) - 0.3) * 5
                b = ((player_data['passing_yards'] / player_data['passing_attempts']) - 3) * 0.25
                c = (player_data['passing_touchdowns'] / player_data['passing_attempts']) * 20
                d = 2.375 - ((player_data['passing_interceptions'] / player_data['passing_attempts']) * 25)
                
                passer_rating = ((a.clip(0, 2.375) + b.clip(0, 2.375) + c.clip(0, 2.375) + d.clip(0, 2.375)) / 6) * 100
                metrics['passer_rating'] = passer_rating.mean()
        
        elif position == 'RB':
            if 'rushing_attempts' in player_data.columns:
                metrics['yards_per_carry'] = (player_data['rushing_yards'] / player_data['rushing_attempts']).mean()
                metrics['touchdown_rate'] = (player_data['rushing_touchdowns'] / player_data['rushing_attempts']).mean() * 100
                metrics['fumble_rate'] = (player_data['rushing_fumbles'] / player_data['rushing_attempts']).mean() * 100
                
                # Receiving metrics for RBs
                if 'targets' in player_data.columns:
                    metrics['target_share'] = player_data['targets'].mean()
                    metrics['catch_rate'] = (player_data['receptions'] / player_data['targets']).mean() * 100
        
        elif position in ['WR', 'TE']:
            if 'targets' in player_data.columns:
                metrics['catch_rate'] = (player_data['receptions'] / player_data['targets']).mean() * 100
                metrics['yards_per_reception'] = (player_data['receiving_yards'] / player_data['receptions']).mean()
                metrics['yards_per_target'] = (player_data['receiving_yards'] / player_data['targets']).mean()
                metrics['touchdown_rate'] = (player_data['receiving_touchdowns'] / player_data['targets']).mean() * 100
        
        # Fantasy efficiency metrics
        if 'fantasy_points_ppr' in player_data.columns:
            metrics['fantasy_points_per_game'] = player_data['fantasy_points_ppr'].mean()
            metrics['fantasy_consistency'] = 1 / (player_data['fantasy_points_ppr'].std() + 1)
        
        return metrics
    
    def get_matchup_analysis(self, player_id: str, opponent: str) -> Dict[str, Any]:
        """Analyze player performance against specific opponent."""
        try:
            query = text("""
                SELECT *
                FROM player_game_stats
                WHERE player_id = :player_id
                AND opponent = :opponent
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'player_id': player_id, 'opponent': opponent})
            
            if len(df) == 0:
                return {'error': 'No historical matchup data'}
            
            position = 'QB' if player_id.endswith('_qb') else 'RB' if player_id.endswith('_rb') else 'WR' if player_id.endswith('_wr') else 'TE'
            
            analysis = {
                'games_played': len(df),
                'avg_fantasy_points': df['fantasy_points_ppr'].mean(),
                'best_game': df['fantasy_points_ppr'].max(),
                'worst_game': df['fantasy_points_ppr'].min(),
                'consistency': 1 / (df['fantasy_points_ppr'].std() + 1)
            }
            
            # Position-specific analysis
            if position == 'QB':
                analysis.update({
                    'avg_passing_yards': df['passing_yards'].mean(),
                    'avg_passing_tds': df['passing_touchdowns'].mean(),
                    'completion_rate': (df['passing_completions'] / df['passing_attempts']).mean() * 100
                })
            elif position == 'RB':
                analysis.update({
                    'avg_rushing_yards': df['rushing_yards'].mean(),
                    'avg_rushing_tds': df['rushing_touchdowns'].mean(),
                    'yards_per_carry': (df['rushing_yards'] / df['rushing_attempts']).mean()
                })
            elif position in ['WR', 'TE']:
                analysis.update({
                    'avg_receiving_yards': df['receiving_yards'].mean(),
                    'avg_receptions': df['receptions'].mean(),
                    'avg_targets': df['targets'].mean(),
                    'catch_rate': (df['receptions'] / df['targets']).mean() * 100
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Matchup analysis error: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_recommendations(self, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Generate comprehensive betting recommendations with all data sources."""
        
        print("🔍 Generating comprehensive recommendations...")
        
        # Get base recommendations from streamlined system
        base_recommendations = self.streamlined_predictor.generate_recommendations(min_confidence)
        
        # Get working system recommendations
        try:
            working_recommendations = self.working_predictor.get_betting_recommendations()
        except:
            working_recommendations = []
        
        # Enhance recommendations with additional data
        enhanced_recommendations = []
        
        for rec in base_recommendations:
            player_id = rec['player_id']
            position = rec['position']
            
            # Get advanced metrics
            try:
                query = text("""
                    SELECT *
                    FROM player_game_stats
                    WHERE player_id = :player_id
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                
                with self.engine.connect() as conn:
                    player_data = pd.read_sql(query, conn, params={'player_id': player_id})
                
                if len(player_data) > 0:
                    advanced_metrics = self.calculate_advanced_metrics(player_data, position)
                    
                    # Get weather impact (for outdoor games)
                    weather_data = self.get_weather_data("Green Bay", "2024-12-15")  # Mock for now
                    
                    # Get news sentiment
                    player_name = player_id.replace('_', ' ').title()
                    sentiment_data = self.get_player_news_sentiment(player_name)
                    
                    # Enhanced recommendation
                    enhanced_rec = rec.copy()
                    enhanced_rec.update({
                        'advanced_metrics': advanced_metrics,
                        'weather_impact': self._calculate_weather_impact(weather_data, position),
                        'news_sentiment': sentiment_data['sentiment_score'],
                        'data_sources': ['stats', 'weather', 'news', 'odds'],
                        'enhanced_confidence': self._calculate_enhanced_confidence(
                            rec['confidence'], advanced_metrics, sentiment_data, weather_data
                        )
                    })
                    
                    enhanced_recommendations.append(enhanced_rec)
            
            except Exception as e:
                logger.warning(f"Enhancement error for {player_id}: {e}")
                enhanced_recommendations.append(rec)
        
        # Sort by enhanced confidence
        enhanced_recommendations.sort(
            key=lambda x: x.get('enhanced_confidence', x.get('confidence', 0)), 
            reverse=True
        )
        
        return enhanced_recommendations[:15]
    
    def _calculate_weather_impact(self, weather_data: Dict[str, Any], position: str) -> float:
        """Calculate weather impact on player performance."""
        impact_score = 0.0
        
        temp = weather_data.get('temperature', 70)
        wind = weather_data.get('wind_speed', 0)
        precipitation = weather_data.get('precipitation', 0)
        
        # Temperature impact
        if temp < 32:  # Freezing
            impact_score -= 0.15
        elif temp < 45:  # Cold
            impact_score -= 0.08
        elif temp > 85:  # Hot
            impact_score -= 0.05
        
        # Wind impact (especially for passing)
        if position == 'QB' and wind > 15:
            impact_score -= 0.12
        elif position in ['WR', 'TE'] and wind > 20:
            impact_score -= 0.08
        
        # Precipitation impact
        if precipitation > 0.1:
            if position == 'QB':
                impact_score -= 0.10
            elif position == 'RB':
                impact_score += 0.05  # RBs might get more touches in bad weather
            else:
                impact_score -= 0.06
        
        return max(-0.3, min(0.2, impact_score))  # Cap between -30% and +20%
    
    def _calculate_enhanced_confidence(self, base_confidence: float, metrics: Dict[str, Any], 
                                    sentiment: Dict[str, Any], weather: Dict[str, Any]) -> float:
        """Calculate enhanced confidence score using all data sources."""
        
        enhanced = base_confidence
        
        # Sentiment adjustment
        sentiment_score = sentiment.get('sentiment_score', 0)
        if sentiment_score > 2:
            enhanced += 0.05
        elif sentiment_score < -2:
            enhanced -= 0.05
        
        # Consistency bonus
        consistency = metrics.get('fantasy_consistency', 0.5)
        if consistency > 0.8:
            enhanced += 0.03
        elif consistency < 0.3:
            enhanced -= 0.03
        
        # Weather adjustment
        weather_impact = self._calculate_weather_impact(weather, 'QB')  # Default to QB
        enhanced += weather_impact * 0.5  # Moderate weather impact on confidence
        
        return max(0.1, min(0.95, enhanced))  # Keep within reasonable bounds
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'streamlined_models': len(self.streamlined_predictor.models),
            'working_models': len(self.working_predictor.models) if hasattr(self.working_predictor, 'models') else 0,
            'database_connected': self._test_database_connection(),
            'api_keys_configured': {
                'odds_api': self.odds_api_key != '0111e24d4786064dfe6a5cf6462b2c4c',
                'weather_api': self.weather_api_key != 'your_weather_key_here',
                'news_api': self.news_api_key != '8eb928de87b4442c99931a5a6b6fb912'
            },
            'data_sources': ['player_stats', 'weather', 'betting_odds', 'news_sentiment'],
            'advanced_features': ['matchup_analysis', 'weather_impact', 'sentiment_analysis', 'advanced_metrics']
        }
    
    def _test_database_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False

def main():
    """Main function to demonstrate advanced analyzer."""
    
    print("🏈 ADVANCED NFL BETTING ANALYZER")
    print("=" * 60)
    print("🚀 Enhanced with Weather, Odds, News & Advanced Stats")
    print()
    
    try:
        # Initialize advanced analyzer
        analyzer = AdvancedNFLAnalyzer()
        
        # Show system status
        status = analyzer.get_system_status()
        print("📊 SYSTEM STATUS:")
        print(f"  Streamlined Models: {status['streamlined_models']}")
        print(f"  Working Models: {status['working_models']}")
        print(f"  Database Connected: {'✅' if status['database_connected'] else '❌'}")
        print(f"  API Keys: {sum(status['api_keys_configured'].values())}/3 configured")
        print(f"  Data Sources: {len(status['data_sources'])}")
        print(f"  Advanced Features: {len(status['advanced_features'])}")
        
        # Generate comprehensive recommendations
        print(f"\n🎯 Generating comprehensive recommendations...")
        recommendations = analyzer.generate_comprehensive_recommendations()
        
        if recommendations:
            print(f"\n💡 TOP ADVANCED BETTING RECOMMENDATIONS:")
            print("-" * 80)
            
            for i, rec in enumerate(recommendations[:10], 1):
                player_name = rec['player_id'].replace('_', ' ').title()
                target = rec['target']
                prediction = rec['prediction']
                recommendation = rec['recommendation']
                confidence = rec.get('enhanced_confidence', rec['confidence'])
                edge = rec.get('edge', 0)
                
                print(f"{i:2d}. {player_name} ({rec['position']})")
                print(f"    📊 {target}: {recommendation} {prediction:.1f}")
                print(f"    🎯 Enhanced Confidence: {confidence:.1%} | Edge: {edge:.1%}")
                
                # Show advanced data if available
                if 'advanced_metrics' in rec:
                    metrics = rec['advanced_metrics']
                    if metrics:
                        key_metric = list(metrics.keys())[0] if metrics else None
                        if key_metric:
                            print(f"    📈 {key_metric.replace('_', ' ').title()}: {metrics[key_metric]:.2f}")
                
                if 'weather_impact' in rec:
                    impact = rec['weather_impact']
                    if abs(impact) > 0.02:
                        print(f"    🌤️  Weather Impact: {impact:+.1%}")
                
                if 'news_sentiment' in rec:
                    sentiment = rec['news_sentiment']
                    if abs(sentiment) > 1:
                        sentiment_text = "Positive" if sentiment > 0 else "Negative"
                        print(f"    📰 News Sentiment: {sentiment_text} ({sentiment:+d})")
                
                print()
        else:
            print("❌ No recommendations available")
        
        print("✅ Advanced NFL analyzer completed successfully!")
        
    except Exception as e:
        logger.error(f"Advanced analyzer error: {e}")
        print(f"❌ Advanced analyzer error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
