"""
Real API Integration for NFL Data Collection
Updated data collector with actual API implementations.
"""

import os
import aiohttp
import requests
import nfl_data_py as nfl
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, date
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RealNFLDataCollector:
    """NFL data collector using real APIs."""
    
    def __init__(self):
        """Initialize with API keys from environment."""
        # API Keys
        self.odds_api_key = os.getenv('ODDS_API_KEY')
        self.weather_api_key = os.getenv('WEATHER_API_KEY') 
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        # API URLs
        self.odds_api_url = "https://api.the-odds-api.com/v4"
        self.weather_api_url = "https://api.openweathermap.org/data/2.5"
        self.news_api_url = "https://newsapi.org/v2"
        self.espn_api_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        
        # Session for HTTP requests
        self.session = requests.Session()
        
    async def collect_nfl_data(self, seasons: List[int]) -> Dict[str, pd.DataFrame]:
        """Collect core NFL data using nfl_data_py (FREE)."""
        logger.info("Collecting NFL data using nfl_data_py...")
        
        data = {}
        
        try:
            # Player stats (FREE)
            logger.info("Downloading weekly stats...")
            data['weekly_stats'] = nfl.import_weekly_data(seasons)
            
            # Schedules (FREE)
            logger.info("Downloading schedules...")
            data['schedules'] = nfl.import_schedules(seasons)
            
            # Rosters (FREE)
            logger.info("Downloading rosters...")
            data['rosters'] = nfl.import_rosters(seasons)
            
            # Injuries (FREE)
            logger.info("Downloading injury data...")
            data['injuries'] = nfl.import_injuries(seasons)
            
            # Play-by-play (FREE - but large)
            logger.info("Downloading play-by-play data...")
            data['pbp'] = nfl.import_pbp_data(seasons)
            
            logger.info("NFL data collection completed successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting NFL data: {e}")
            raise
            
    async def collect_betting_odds(self) -> List[Dict]:
        """Collect betting odds from The Odds API."""
        if not self.odds_api_key:
            logger.warning("No Odds API key provided, skipping betting data")
            return []
            
        logger.info("Collecting betting odds...")
        
        odds_data = []
        
        try:
            # Get available sports
            sports_url = f"{self.odds_api_url}/sports"
            params = {
                'api_key': self.odds_api_key
            }
            
            response = self.session.get(sports_url, params=params)
            response.raise_for_status()
            
            # Find NFL sport key
            sports = response.json()
            nfl_sport = None
            for sport in sports:
                if 'americanfootball_nfl' in sport.get('key', ''):
                    nfl_sport = sport['key']
                    break
                    
            if not nfl_sport:
                logger.warning("NFL not found in available sports")
                return []
                
            # Get NFL odds
            odds_url = f"{self.odds_api_url}/sports/{nfl_sport}/odds"
            params = {
                'api_key': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = self.session.get(odds_url, params=params)
            response.raise_for_status()
            
            odds_data = response.json()
            logger.info(f"Collected odds for {len(odds_data)} games")
            
            # Get player props if available
            props_data = await self._collect_player_props(nfl_sport)
            
            return {
                'game_odds': odds_data,
                'player_props': props_data
            }
            
        except Exception as e:
            logger.error(f"Error collecting betting odds: {e}")
            return []
            
    async def _collect_player_props(self, sport_key: str) -> List[Dict]:
        """Collect player prop bets."""
        try:
            props_url = f"{self.odds_api_url}/sports/{sport_key}/odds"
            params = {
                'api_key': self.odds_api_key,
                'regions': 'us',
                'markets': 'player_pass_yds,player_rush_yds,player_receptions',
                'oddsFormat': 'american'
            }
            
            response = self.session.get(props_url, params=params)
            response.raise_for_status()
            
            props_data = response.json()
            logger.info(f"Collected props for {len(props_data)} games")
            
            return props_data
            
        except Exception as e:
            logger.warning(f"Error collecting player props: {e}")
            return []
            
    async def collect_weather_data(self, games: List[Dict]) -> Dict[str, Dict]:
        """Collect weather data for games."""
        if not self.weather_api_key:
            logger.warning("No Weather API key provided, skipping weather data")
            return {}
            
        logger.info("Collecting weather data...")
        
        weather_data = {}
        
        # Stadium locations (you'd have a more complete database)
        stadium_locations = {
            'Arrowhead Stadium': {'lat': 39.0489, 'lon': -94.4839},
            'Highmark Stadium': {'lat': 42.7738, 'lon': -78.7870},
            'Lambeau Field': {'lat': 44.5013, 'lon': -88.0622},
            'Soldier Field': {'lat': 41.8623, 'lon': -87.6167},
            # Add more stadiums as needed
        }
        
        try:
            for game in games:
                stadium = game.get('stadium')
                game_date = game.get('game_date')
                
                if stadium in stadium_locations and game_date:
                    location = stadium_locations[stadium]
                    
                    # Get weather forecast/current conditions
                    weather_url = f"{self.weather_api_url}/weather"
                    params = {
                        'lat': location['lat'],
                        'lon': location['lon'],
                        'appid': self.weather_api_key,
                        'units': 'imperial'
                    }
                    
                    response = self.session.get(weather_url, params=params)
                    response.raise_for_status()
                    
                    weather_info = response.json()
                    
                    weather_data[game['game_id']] = {
                        'temperature': weather_info['main']['temp'],
                        'humidity': weather_info['main']['humidity'],
                        'wind_speed': weather_info['wind']['speed'],
                        'conditions': weather_info['weather'][0]['description'],
                        'visibility': weather_info.get('visibility', 10000)
                    }
                    
            logger.info(f"Collected weather data for {len(weather_data)} games")
            return weather_data
            
        except Exception as e:
            logger.error(f"Error collecting weather data: {e}")
            return {}
            
    async def collect_news_and_injuries(self) -> List[Dict]:
        """Collect NFL news and injury reports."""
        logger.info("Collecting news and injury data...")
        
        news_data = []
        
        try:
            # Method 1: NewsAPI (if key available)
            if self.news_api_key:
                news_data.extend(await self._collect_from_newsapi())
                
            # Method 2: ESPN API (free, no key needed)
            news_data.extend(await self._collect_from_espn())
            
            logger.info(f"Collected {len(news_data)} news articles")
            return news_data
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return []
            
    async def _collect_from_newsapi(self) -> List[Dict]:
        """Collect from NewsAPI."""
        try:
            news_url = f"{self.news_api_url}/everything"
            params = {
                'apiKey': self.news_api_key,
                'q': 'NFL injury OR "injury report"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100
            }
            
            response = self.session.get(news_url, params=params)
            response.raise_for_status()
            
            return response.json()['articles']
            
        except Exception as e:
            logger.warning(f"NewsAPI collection failed: {e}")
            return []
            
    async def _collect_from_espn(self) -> List[Dict]:
        """Collect from ESPN API (free)."""
        try:
            # ESPN NFL news endpoint
            news_url = f"{self.espn_api_url}/news"
            
            response = self.session.get(news_url)
            response.raise_for_status()
            
            news_data = response.json()
            return news_data.get('articles', [])
            
        except Exception as e:
            logger.warning(f"ESPN API collection failed: {e}")
            return []
            
    async def collect_live_scores(self) -> Dict[str, Any]:
        """Collect live scores and game status."""
        logger.info("Collecting live scores...")
        
        try:
            # ESPN scoreboard (free)
            scoreboard_url = f"{self.espn_api_url}/scoreboard"
            
            response = self.session.get(scoreboard_url)
            response.raise_for_status()
            
            scoreboard_data = response.json()
            
            games = []
            for game in scoreboard_data.get('events', []):
                game_info = {
                    'game_id': game['id'],
                    'status': game['status']['type']['description'],
                    'home_team': game['competitions'][0]['competitors'][0]['team']['abbreviation'],
                    'away_team': game['competitions'][0]['competitors'][1]['team']['abbreviation'],
                    'home_score': game['competitions'][0]['competitors'][0].get('score', 0),
                    'away_score': game['competitions'][0]['competitors'][1].get('score', 0),
                    'game_date': game['date']
                }
                games.append(game_info)
                
            logger.info(f"Collected live data for {len(games)} games")
            return {'games': games}
            
        except Exception as e:
            logger.error(f"Error collecting live scores: {e}")
            return {'games': []}


# Usage example
async def example_usage():
    """Example of how to use the real API collector."""
    
    collector = RealNFLDataCollector()
    
    # 1. Collect core NFL data (FREE)
    nfl_data = await collector.collect_nfl_data([2024])
    print(f"Collected weekly stats: {len(nfl_data['weekly_stats'])} records")
    
    # 2. Collect betting odds (requires API key)
    betting_data = await collector.collect_betting_odds()
    print(f"Collected betting data: {len(betting_data)} games")
    
    # 3. Collect weather (requires API key)
    sample_games = [
        {'game_id': '2024_10_KC_BUF', 'stadium': 'Arrowhead Stadium', 'game_date': '2024-11-10'}
    ]
    weather_data = await collector.collect_weather_data(sample_games)
    print(f"Collected weather for {len(weather_data)} games")
    
    # 4. Collect news
    news_data = await collector.collect_news_and_injuries()
    print(f"Collected {len(news_data)} news articles")
    
    # 5. Collect live scores (FREE)
    live_data = await collector.collect_live_scores()
    print(f"Collected live data for {len(live_data['games'])} games")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())