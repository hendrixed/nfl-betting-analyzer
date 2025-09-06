"""
Real-Time Data Integration for NFL Betting Analyzer
Integrates live odds, injury reports, weather data, and market movements
"""

import asyncio
import aiohttp
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import time
from dataclasses import dataclass
import sqlite3
from sqlalchemy import create_engine, text
import os
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class InjuryReport:
    player_id: str
    injury_status: str  # 'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'PROBABLE'
    injury_type: str
    updated_at: datetime

@dataclass
class WeatherData:
    game_id: str
    temperature: float
    wind_speed: float
    precipitation: float
    conditions: str
    dome: bool

@dataclass
class LiveOdds:
    player_id: str
    stat_type: str
    over_under_line: float
    over_odds: int
    under_odds: int
    sportsbook: str
    updated_at: datetime

class RealTimeDataIntegrator:
    """Integrates real-time data sources for enhanced predictions."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.session = aiohttp.ClientSession()
        
        # API endpoints (using placeholder URLs - replace with actual APIs)
        self.api_endpoints = {
            'injuries': 'https://api.sportsdata.io/v3/nfl/scores/json/Injuries',
            'weather': 'https://api.openweathermap.org/data/2.5/weather',
            'odds': 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds',
            'player_props': 'https://api.draftkings.com/sites/US-DK/api/v5/eventgroups/88808'
        }
        
        # Cache for real-time data
        self.injury_cache = {}
        self.weather_cache = {}
        self.odds_cache = {}
        
        # Update intervals (minutes)
        self.update_intervals = {
            'injuries': 30,
            'weather': 60,
            'odds': 5
        }
        
        self.last_updates = {}
    
    async def get_injury_reports(self) -> List[InjuryReport]:
        """Fetch current injury reports."""
        try:
            # Check if we need to update
            if self._should_update('injuries'):
                headers = {'Ocp-Apim-Subscription-Key': self.config.sports_data_api_key}
                
                async with self.session.get(
                    self.api_endpoints['injuries'], 
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        injury_reports = []
                        for injury in data:
                            report = InjuryReport(
                                player_id=f"{injury.get('Name', '').lower().replace(' ', '')}_{injury.get('Position', '').lower()}",
                                injury_status=injury.get('Status', 'UNKNOWN'),
                                injury_type=injury.get('BodyPart', 'Unknown'),
                                updated_at=datetime.now()
                            )
                            injury_reports.append(report)
                        
                        self.injury_cache = {r.player_id: r for r in injury_reports}
                        self.last_updates['injuries'] = datetime.now()
                        
                        logger.info(f"Updated {len(injury_reports)} injury reports")
                        return injury_reports
            
            # Return cached data
            return list(self.injury_cache.values())
            
        except Exception as e:
            logger.error(f"Error fetching injury reports: {e}")
            return []
    
    async def get_weather_data(self, game_locations: List[str]) -> List[WeatherData]:
        """Fetch weather data for game locations."""
        try:
            if self._should_update('weather'):
                weather_data = []
                
                for location in game_locations:
                    params = {
                        'q': location,
                        'appid': self.config.weather_api_key,
                        'units': 'imperial'
                    }
                    
                    async with self.session.get(
                        self.api_endpoints['weather'],
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            weather = WeatherData(
                                game_id=location,
                                temperature=data['main']['temp'],
                                wind_speed=data['wind']['speed'],
                                precipitation=data.get('rain', {}).get('1h', 0),
                                conditions=data['weather'][0]['description'],
                                dome=self._is_dome_stadium(location)
                            )
                            weather_data.append(weather)
                
                self.weather_cache = {w.game_id: w for w in weather_data}
                self.last_updates['weather'] = datetime.now()
                
                logger.info(f"Updated weather for {len(weather_data)} locations")
                return weather_data
            
            return list(self.weather_cache.values())
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return []
    
    async def get_live_odds(self) -> List[LiveOdds]:
        """Fetch live betting odds and player props."""
        try:
            if self._should_update('odds'):
                headers = {'X-RapidAPI-Key': self.config.odds_api_key}
                
                async with self.session.get(
                    self.api_endpoints['odds'],
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        live_odds = []
                        for game in data:
                            for bookmaker in game.get('bookmakers', []):
                                for market in bookmaker.get('markets', []):
                                    if market['key'] in ['player_pass_yds', 'player_rush_yds', 'player_rec_yds']:
                                        for outcome in market['outcomes']:
                                            odds = LiveOdds(
                                                player_id=outcome.get('description', '').lower().replace(' ', '_'),
                                                stat_type=market['key'],
                                                over_under_line=outcome.get('point', 0),
                                                over_odds=outcome.get('price', 0),
                                                under_odds=0,  # Would need to find corresponding under
                                                sportsbook=bookmaker['title'],
                                                updated_at=datetime.now()
                                            )
                                            live_odds.append(odds)
                        
                        self.odds_cache = {f"{o.player_id}_{o.stat_type}": o for o in live_odds}
                        self.last_updates['odds'] = datetime.now()
                        
                        logger.info(f"Updated {len(live_odds)} live odds")
                        return live_odds
            
            return list(self.odds_cache.values())
            
        except Exception as e:
            logger.error(f"Error fetching live odds: {e}")
            return []
    
    def _should_update(self, data_type: str) -> bool:
        """Check if data should be updated based on interval."""
        if data_type not in self.last_updates:
            return True
        
        time_since_update = datetime.now() - self.last_updates[data_type]
        return time_since_update.total_seconds() > (self.update_intervals[data_type] * 60)
    
    def _is_dome_stadium(self, location: str) -> bool:
        """Check if stadium is a dome (affects weather impact)."""
        dome_stadiums = {
            'atlanta', 'detroit', 'houston', 'indianapolis', 'las vegas',
            'los angeles', 'minnesota', 'new orleans', 'arizona'
        }
        return any(city in location.lower() for city in dome_stadiums)
    
    def get_injury_impact_factor(self, player_id: str) -> float:
        """Get injury impact factor for predictions."""
        if player_id in self.injury_cache:
            injury = self.injury_cache[player_id]
            
            impact_factors = {
                'OUT': 0.0,
                'DOUBTFUL': 0.3,
                'QUESTIONABLE': 0.7,
                'PROBABLE': 0.9,
                'UNKNOWN': 1.0
            }
            
            return impact_factors.get(injury.injury_status, 1.0)
        
        return 1.0  # No injury data, assume healthy
    
    def get_weather_impact_factor(self, game_location: str, position: str) -> float:
        """Get weather impact factor for predictions."""
        if game_location in self.weather_cache:
            weather = self.weather_cache[game_location]
            
            # Dome games not affected by weather
            if weather.dome:
                return 1.0
            
            impact = 1.0
            
            # Temperature effects
            if weather.temperature < 32:  # Freezing
                if position == 'QB':
                    impact *= 0.9  # Passing accuracy decreases
                elif position in ['WR', 'TE']:
                    impact *= 0.85  # Catching becomes harder
            
            # Wind effects (primarily passing)
            if weather.wind_speed > 15:
                if position == 'QB':
                    impact *= 0.8
                elif position in ['WR', 'TE']:
                    impact *= 0.9
            
            # Precipitation effects
            if weather.precipitation > 0.1:
                if position == 'QB':
                    impact *= 0.85
                elif position in ['WR', 'TE']:
                    impact *= 0.8
                elif position == 'RB':
                    impact *= 1.1  # More rushing in bad weather
            
            return max(impact, 0.5)  # Minimum 50% of normal performance
        
        return 1.0  # No weather data available
    
    def get_market_value_score(self, player_id: str, stat_type: str, predicted_value: float) -> float:
        """Calculate value score based on market odds."""
        odds_key = f"{player_id}_{stat_type}"
        
        if odds_key in self.odds_cache:
            odds = self.odds_cache[odds_key]
            
            # Simple value calculation
            # If our prediction is significantly higher than the line, it's good value
            if predicted_value > odds.over_under_line * 1.1:
                return 1.2  # High value bet
            elif predicted_value > odds.over_under_line:
                return 1.0  # Fair value
            else:
                return 0.8  # Poor value
        
        return 1.0  # No market data available
    
    async def close(self):
        """Close the aiohttp session."""
        await self.session.close()


class MarketMovementTracker:
    """Track and analyze betting market movements."""
    
    def __init__(self, db_path: str = "data/market_movements.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize market movement tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                stat_type TEXT,
                sportsbook TEXT,
                line_value REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                timestamp DATETIME,
                INDEX(player_id, stat_type, timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_market_data(self, odds_data: List[LiveOdds]):
        """Record current market data for movement tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for odds in odds_data:
            cursor.execute("""
                INSERT INTO market_movements 
                (player_id, stat_type, sportsbook, line_value, over_odds, under_odds, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                odds.player_id, odds.stat_type, odds.sportsbook,
                odds.over_under_line, odds.over_odds, odds.under_odds,
                odds.updated_at
            ))
        
        conn.commit()
        conn.close()
    
    def get_line_movement(self, player_id: str, stat_type: str, hours_back: int = 24) -> Dict:
        """Get line movement analysis for a player/stat."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT line_value, over_odds, under_odds, timestamp
            FROM market_movements
            WHERE player_id = ? AND stat_type = ?
            AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        """.format(hours_back)
        
        df = pd.read_sql_query(query, conn, params=(player_id, stat_type))
        conn.close()
        
        if df.empty:
            return {}
        
        # Calculate movement metrics
        initial_line = df.iloc[0]['line_value']
        current_line = df.iloc[-1]['line_value']
        line_movement = current_line - initial_line
        
        # Odds movement
        initial_over_odds = df.iloc[0]['over_odds']
        current_over_odds = df.iloc[-1]['over_odds']
        
        return {
            'initial_line': initial_line,
            'current_line': current_line,
            'line_movement': line_movement,
            'movement_percentage': (line_movement / initial_line * 100) if initial_line != 0 else 0,
            'initial_over_odds': initial_over_odds,
            'current_over_odds': current_over_odds,
            'sharp_money_indicator': self._detect_sharp_money(df)
        }
    
    def _detect_sharp_money(self, df: pd.DataFrame) -> str:
        """Detect if sharp money is moving the line."""
        if len(df) < 2:
            return "INSUFFICIENT_DATA"
        
        # Simple heuristic: line moves against public betting percentage
        line_movement = df.iloc[-1]['line_value'] - df.iloc[0]['line_value']
        odds_movement = df.iloc[-1]['over_odds'] - df.iloc[0]['over_odds']
        
        # If line moves up but odds get worse (higher), suggests sharp money on over
        if line_movement > 0.5 and odds_movement > 10:
            return "SHARP_MONEY_OVER"
        elif line_movement < -0.5 and odds_movement < -10:
            return "SHARP_MONEY_UNDER"
        else:
            return "PUBLIC_MONEY"


class EnhancedPredictionAdjuster:
    """Adjust predictions based on real-time factors."""
    
    def __init__(self, data_integrator: RealTimeDataIntegrator, 
                 movement_tracker: MarketMovementTracker):
        self.data_integrator = data_integrator
        self.movement_tracker = movement_tracker
    
    def adjust_prediction(self, player_id: str, position: str, base_prediction: float,
                         stat_type: str, game_location: str = None) -> Dict:
        """Adjust base prediction with real-time factors."""
        
        adjusted_prediction = base_prediction
        adjustment_factors = {}
        
        # Injury adjustment
        injury_factor = self.data_integrator.get_injury_impact_factor(player_id)
        adjusted_prediction *= injury_factor
        adjustment_factors['injury'] = injury_factor
        
        # Weather adjustment
        if game_location:
            weather_factor = self.data_integrator.get_weather_impact_factor(game_location, position)
            adjusted_prediction *= weather_factor
            adjustment_factors['weather'] = weather_factor
        
        # Market value assessment
        market_value = self.data_integrator.get_market_value_score(player_id, stat_type, adjusted_prediction)
        adjustment_factors['market_value'] = market_value
        
        # Line movement analysis
        line_movement = self.movement_tracker.get_line_movement(player_id, stat_type)
        if line_movement:
            # Adjust confidence based on sharp money indicators
            sharp_indicator = line_movement.get('sharp_money_indicator', 'PUBLIC_MONEY')
            if sharp_indicator in ['SHARP_MONEY_OVER', 'SHARP_MONEY_UNDER']:
                adjustment_factors['sharp_money'] = sharp_indicator
        
        return {
            'original_prediction': base_prediction,
            'adjusted_prediction': adjusted_prediction,
            'adjustment_factors': adjustment_factors,
            'confidence_modifier': self._calculate_confidence_modifier(adjustment_factors),
            'betting_recommendation': self._generate_betting_recommendation(
                adjusted_prediction, adjustment_factors, line_movement
            )
        }
    
    def _calculate_confidence_modifier(self, factors: Dict) -> float:
        """Calculate confidence modifier based on adjustment factors."""
        modifier = 1.0
        
        # Injury reduces confidence
        if factors.get('injury', 1.0) < 0.8:
            modifier *= 0.8
        
        # Bad weather reduces confidence
        if factors.get('weather', 1.0) < 0.9:
            modifier *= 0.9
        
        # Sharp money increases confidence
        if factors.get('sharp_money') in ['SHARP_MONEY_OVER', 'SHARP_MONEY_UNDER']:
            modifier *= 1.2
        
        return min(modifier, 1.5)  # Cap at 50% increase
    
    def _generate_betting_recommendation(self, prediction: float, factors: Dict, 
                                       line_movement: Dict) -> str:
        """Generate specific betting recommendation."""
        recommendations = []
        
        # Market value recommendation
        market_value = factors.get('market_value', 1.0)
        if market_value > 1.1:
            recommendations.append("STRONG BET - High market value detected")
        elif market_value < 0.9:
            recommendations.append("AVOID - Poor market value")
        
        # Sharp money recommendation
        sharp_money = factors.get('sharp_money')
        if sharp_money == 'SHARP_MONEY_OVER':
            recommendations.append("FOLLOW SHARP MONEY - Bet Over")
        elif sharp_money == 'SHARP_MONEY_UNDER':
            recommendations.append("FOLLOW SHARP MONEY - Bet Under")
        
        # Injury recommendation
        injury_factor = factors.get('injury', 1.0)
        if injury_factor < 0.5:
            recommendations.append("AVOID - Significant injury concern")
        
        return " | ".join(recommendations) if recommendations else "STANDARD BET"


# Example usage and testing
async def main():
    """Example usage of real-time data integration."""
    
    config = ConfigManager()
    integrator = RealTimeDataIntegrator(config)
    tracker = MarketMovementTracker()
    adjuster = EnhancedPredictionAdjuster(integrator, tracker)
    
    try:
        # Fetch real-time data
        print("🔄 Fetching real-time data...")
        
        injuries = await integrator.get_injury_reports()
        print(f"📋 Injury reports: {len(injuries)}")
        
        weather = await integrator.get_weather_data(['New York', 'Los Angeles'])
        print(f"🌤️  Weather data: {len(weather)}")
        
        odds = await integrator.get_live_odds()
        print(f"💰 Live odds: {len(odds)}")
        
        # Record market data
        if odds:
            tracker.record_market_data(odds)
            print("📊 Market data recorded")
        
        # Example prediction adjustment
        adjusted = adjuster.adjust_prediction(
            player_id="pmahomes_qb",
            position="QB", 
            base_prediction=285.5,
            stat_type="passing_yards",
            game_location="Kansas City"
        )
        
        print("\n🎯 Example Prediction Adjustment:")
        print(f"   Original: {adjusted['original_prediction']}")
        print(f"   Adjusted: {adjusted['adjusted_prediction']:.1f}")
        print(f"   Factors: {adjusted['adjustment_factors']}")
        print(f"   Recommendation: {adjusted['betting_recommendation']}")
        
    finally:
        await integrator.close()


if __name__ == "__main__":
    asyncio.run(main())
