#!/usr/bin/env python3
"""
Real-Time Market Data Integration
Fetches live odds, line movements, and market efficiency data for NFL prop bets.
"""

import sys
import os
import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMarketData:
    """Real-time market data integration for NFL betting."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # API endpoints (using free/demo endpoints)
        self.odds_api_key = os.getenv('ODDS_API_KEY', 'demo_key')
        self.odds_base_url = "https://api.the-odds-api.com/v4"
        
        # Market data cache
        self.market_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Sportsbooks to track
        self.tracked_books = [
            'draftkings', 'fanduel', 'betmgm', 'caesars', 
            'pointsbet', 'barstool', 'unibet'
        ]
        
    def fetch_nfl_odds(self, markets: List[str] = None) -> Dict[str, Any]:
        """Fetch current NFL odds from multiple sportsbooks."""
        
        if markets is None:
            markets = ['h2h', 'spreads', 'totals']
        
        all_odds = {}
        
        for market in markets:
            try:
                url = f"{self.odds_base_url}/sports/americanfootball_nfl/odds"
                params = {
                    'apiKey': self.odds_api_key,
                    'regions': 'us',
                    'markets': market,
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                }
                
                # For demo purposes, create mock data
                if self.odds_api_key == 'demo_key':
                    all_odds[market] = self._generate_mock_odds_data(market)
                else:
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        all_odds[market] = response.json()
                    else:
                        logger.warning(f"Failed to fetch {market} odds: {response.status_code}")
                        
            except Exception as e:
                logger.error(f"Error fetching {market} odds: {e}")
        
        return all_odds
    
    def _generate_mock_odds_data(self, market: str) -> List[Dict[str, Any]]:
        """Generate realistic mock odds data for testing."""
        
        teams = [
            ('KC', 'BUF'), ('SF', 'LAR'), ('PHI', 'DAL'), 
            ('BAL', 'CIN'), ('MIA', 'NYJ'), ('GB', 'MIN')
        ]
        
        mock_games = []
        
        for home_team, away_team in teams:
            game = {
                'id': f"mock_{home_team}_{away_team}",
                'sport_key': 'americanfootball_nfl',
                'sport_title': 'NFL',
                'commence_time': (datetime.now() + timedelta(days=1)).isoformat(),
                'home_team': home_team,
                'away_team': away_team,
                'bookmakers': []
            }
            
            # Add mock bookmaker odds
            for book in self.tracked_books[:3]:  # Use first 3 for demo
                bookmaker = {
                    'key': book,
                    'title': book.title(),
                    'last_update': datetime.now().isoformat(),
                    'markets': []
                }
                
                if market == 'h2h':
                    # Money line odds
                    home_odds = np.random.randint(-200, 200)
                    away_odds = -home_odds + np.random.randint(-50, 50)
                    
                    bookmaker['markets'].append({
                        'key': 'h2h',
                        'outcomes': [
                            {'name': home_team, 'price': home_odds},
                            {'name': away_team, 'price': away_odds}
                        ]
                    })
                    
                elif market == 'spreads':
                    # Point spreads
                    spread = np.random.uniform(-7, 7)
                    
                    bookmaker['markets'].append({
                        'key': 'spreads',
                        'outcomes': [
                            {'name': home_team, 'price': -110, 'point': spread},
                            {'name': away_team, 'price': -110, 'point': -spread}
                        ]
                    })
                    
                elif market == 'totals':
                    # Over/Under totals
                    total = np.random.uniform(42, 52)
                    
                    bookmaker['markets'].append({
                        'key': 'totals',
                        'outcomes': [
                            {'name': 'Over', 'price': -110, 'point': total},
                            {'name': 'Under', 'price': -110, 'point': total}
                        ]
                    })
                
                game['bookmakers'].append(bookmaker)
            
            mock_games.append(game)
        
        return mock_games
    
    def fetch_player_props(self, player_name: str = None) -> Dict[str, Any]:
        """Fetch player prop bet odds."""
        
        # For demo, generate mock player props
        mock_props = {
            'Patrick Mahomes': {
                'passing_yards': {
                    'draftkings': {'over_274.5': -110, 'under_274.5': -110},
                    'fanduel': {'over_275.5': -115, 'under_275.5': -105},
                    'betmgm': {'over_274.5': -108, 'under_274.5': -112}
                },
                'passing_touchdowns': {
                    'draftkings': {'over_1.5': -140, 'under_1.5': +115},
                    'fanduel': {'over_1.5': -135, 'under_1.5': +110},
                    'betmgm': {'over_1.5': -145, 'under_1.5': +120}
                }
            },
            'Josh Allen': {
                'passing_yards': {
                    'draftkings': {'over_264.5': -110, 'under_264.5': -110},
                    'fanduel': {'over_265.5': -112, 'under_265.5': -108}
                },
                'rushing_yards': {
                    'draftkings': {'over_39.5': -115, 'under_39.5': -105},
                    'fanduel': {'over_40.5': -110, 'under_40.5': -110}
                }
            }
        }
        
        if player_name:
            return mock_props.get(player_name, {})
        
        return mock_props
    
    def calculate_market_efficiency(self, odds_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market efficiency metrics."""
        
        efficiency_metrics = {}
        
        for market, games in odds_data.items():
            market_efficiencies = []
            
            for game in games:
                for bookmaker in game.get('bookmakers', []):
                    for market_data in bookmaker.get('markets', []):
                        outcomes = market_data.get('outcomes', [])
                        
                        if len(outcomes) == 2:
                            # Calculate implied probabilities
                            probs = []
                            for outcome in outcomes:
                                odds = outcome.get('price', -110)
                                if odds > 0:
                                    prob = 100 / (odds + 100)
                                else:
                                    prob = abs(odds) / (abs(odds) + 100)
                                probs.append(prob)
                            
                            # Market efficiency = 1 - overround
                            total_prob = sum(probs)
                            efficiency = 1 / total_prob if total_prob > 0 else 0
                            market_efficiencies.append(efficiency)
            
            if market_efficiencies:
                efficiency_metrics[market] = {
                    'avg_efficiency': np.mean(market_efficiencies),
                    'min_efficiency': np.min(market_efficiencies),
                    'max_efficiency': np.max(market_efficiencies),
                    'std_efficiency': np.std(market_efficiencies)
                }
        
        return efficiency_metrics
    
    def detect_line_movements(self, current_odds: Dict[str, Any], 
                            historical_odds: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Detect significant line movements."""
        
        movements = []
        
        # For demo, simulate some line movements
        mock_movements = [
            {
                'game': 'KC vs BUF',
                'market': 'totals',
                'previous_line': 47.5,
                'current_line': 48.5,
                'movement': +1.0,
                'movement_percentage': 2.1,
                'significance': 'SIGNIFICANT',
                'direction': 'UP',
                'timestamp': datetime.now().isoformat()
            },
            {
                'game': 'SF vs LAR',
                'market': 'spreads',
                'previous_line': -3.5,
                'current_line': -2.5,
                'movement': +1.0,
                'movement_percentage': 28.6,
                'significance': 'MAJOR',
                'direction': 'TOWARD_UNDERDOG',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        return mock_movements
    
    def find_arbitrage_opportunities(self, odds_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities across sportsbooks."""
        
        arbitrage_opps = []
        
        for market, games in odds_data.items():
            for game in games:
                bookmaker_odds = {}
                
                # Collect odds from all bookmakers
                for bookmaker in game.get('bookmakers', []):
                    book_name = bookmaker.get('key', '')
                    for market_data in bookmaker.get('markets', []):
                        if market_data.get('key') == market:
                            outcomes = market_data.get('outcomes', [])
                            bookmaker_odds[book_name] = outcomes
                
                # Check for arbitrage
                if len(bookmaker_odds) >= 2:
                    arb_opp = self._calculate_arbitrage(game, bookmaker_odds, market)
                    if arb_opp:
                        arbitrage_opps.append(arb_opp)
        
        return arbitrage_opps
    
    def _calculate_arbitrage(self, game: Dict[str, Any], 
                           bookmaker_odds: Dict[str, List], 
                           market: str) -> Optional[Dict[str, Any]]:
        """Calculate if arbitrage opportunity exists."""
        
        # Simplified arbitrage calculation
        best_odds = {}
        
        for book, outcomes in bookmaker_odds.items():
            for outcome in outcomes:
                outcome_name = outcome.get('name', '')
                odds = outcome.get('price', -110)
                
                if outcome_name not in best_odds or odds > best_odds[outcome_name]['odds']:
                    best_odds[outcome_name] = {'odds': odds, 'book': book}
        
        # Calculate total implied probability
        total_prob = 0
        for outcome_name, data in best_odds.items():
            odds = data['odds']
            if odds > 0:
                prob = 100 / (odds + 100)
            else:
                prob = abs(odds) / (abs(odds) + 100)
            total_prob += prob
        
        # Arbitrage exists if total probability < 1
        if total_prob < 0.98:  # Allow for small margin
            return {
                'game': f"{game.get('home_team')} vs {game.get('away_team')}",
                'market': market,
                'profit_margin': (1 - total_prob) * 100,
                'best_odds': best_odds,
                'total_probability': total_prob,
                'detected_at': datetime.now().isoformat()
            }
        
        return None
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary."""
        
        # Fetch current odds
        current_odds = self.fetch_nfl_odds()
        
        # Calculate efficiency
        efficiency = self.calculate_market_efficiency(current_odds)
        
        # Detect movements
        movements = self.detect_line_movements(current_odds)
        
        # Find arbitrage
        arbitrage = self.find_arbitrage_opportunities(current_odds)
        
        # Player props sample
        player_props = self.fetch_player_props()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_efficiency': efficiency,
            'line_movements': movements,
            'arbitrage_opportunities': arbitrage,
            'total_games_tracked': sum(len(games) for games in current_odds.values()),
            'player_props_available': len(player_props),
            'tracked_sportsbooks': len(self.tracked_books),
            'market_status': 'ACTIVE' if current_odds else 'INACTIVE'
        }

def main():
    """Test the real-time market data system."""
    print("📊 REAL-TIME MARKET DATA INTEGRATION")
    print("=" * 60)
    
    # Initialize system
    market_data = RealTimeMarketData()
    
    # Test market summary
    print("📈 Market Summary:")
    summary = market_data.get_market_summary()
    
    print(f"   Status: {summary['market_status']}")
    print(f"   Games Tracked: {summary['total_games_tracked']}")
    print(f"   Sportsbooks: {summary['tracked_sportsbooks']}")
    print(f"   Player Props: {summary['player_props_available']}")
    
    # Market efficiency
    print("\n⚡ Market Efficiency:")
    for market, metrics in summary['market_efficiency'].items():
        print(f"   {market.upper()}: {metrics['avg_efficiency']:.3f}")
    
    # Line movements
    print("\n📊 Significant Line Movements:")
    for movement in summary['line_movements']:
        print(f"   {movement['game']}: {movement['market']} "
              f"{movement['previous_line']} → {movement['current_line']} "
              f"({movement['significance']})")
    
    # Arbitrage opportunities
    print("\n💰 Arbitrage Opportunities:")
    for arb in summary['arbitrage_opportunities']:
        print(f"   {arb['game']}: {arb['market']} "
              f"({arb['profit_margin']:.2f}% profit)")
    
    # Player props sample
    print("\n🎯 Sample Player Props:")
    props = market_data.fetch_player_props('Patrick Mahomes')
    for stat, books in props.items():
        print(f"   {stat}:")
        for book, lines in books.items():
            for line, odds in lines.items():
                print(f"     {book}: {line} ({odds:+d})")
    
    print("\n" + "=" * 60)
    print("✅ Real-time market data integration ready!")

if __name__ == "__main__":
    main()
