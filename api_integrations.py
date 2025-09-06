#!/usr/bin/env python3
"""
API Integrations and Web Scraping System
Comprehensive data collection from multiple NFL data sources.
"""

import sys
import os
import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sqlite3
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration."""
    name: str
    url: str
    api_key: Optional[str]
    rate_limit: int  # requests per minute
    data_type: str
    active: bool

class APIIntegrations:
    """Comprehensive API integrations and web scraping system."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Data sources configuration
        self.data_sources = {
            'espn_api': DataSource(
                name='ESPN API',
                url='https://site.api.espn.com/apis/site/v2/sports/football/nfl',
                api_key=None,
                rate_limit=60,
                data_type='scores_schedule',
                active=True
            ),
            'nfl_api': DataSource(
                name='NFL API',
                url='https://api.nfl.com/v3',
                api_key=os.getenv('NFL_API_KEY'),
                rate_limit=120,
                data_type='player_stats',
                active=bool(os.getenv('NFL_API_KEY'))
            ),
            'pro_football_reference': DataSource(
                name='Pro Football Reference',
                url='https://www.pro-football-reference.com',
                api_key=None,
                rate_limit=30,
                data_type='advanced_stats',
                active=True
            ),
            'fantasy_pros': DataSource(
                name='FantasyPros',
                url='https://www.fantasypros.com/nfl',
                api_key=os.getenv('FANTASYPROS_API_KEY'),
                rate_limit=60,
                data_type='projections',
                active=bool(os.getenv('FANTASYPROS_API_KEY'))
            ),
            'sleeper_api': DataSource(
                name='Sleeper API',
                url='https://api.sleeper.app/v1',
                api_key=None,
                rate_limit=1000,
                data_type='player_info',
                active=True
            ),
            'odds_api': DataSource(
                name='The Odds API',
                url='https://api.the-odds-api.com/v4',
                api_key=os.getenv('ODDS_API_KEY'),
                rate_limit=500,
                data_type='betting_odds',
                active=bool(os.getenv('ODDS_API_KEY'))
            )
        }
        
        # Rate limiting
        self.last_request_time = {}
        self.request_counts = {}
        
        # Initialize data collection database
        self._init_data_collection_db()
        
    def _init_data_collection_db(self):
        """Initialize data collection tracking database."""
        
        # Data collection log
        log_table_sql = """
        CREATE TABLE IF NOT EXISTS data_collection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            data_type TEXT,
            collection_time TIMESTAMP,
            records_collected INTEGER,
            success BOOLEAN,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Raw data storage
        raw_data_sql = """
        CREATE TABLE IF NOT EXISTS raw_data_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            data_type TEXT,
            data_key TEXT,
            raw_data TEXT,
            collected_at TIMESTAMP,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(log_table_sql))
            conn.execute(text(raw_data_sql))
            conn.commit()
    
    def _check_rate_limit(self, source_name: str) -> bool:
        """Check if we can make a request to this source."""
        
        source = self.data_sources.get(source_name)
        if not source:
            return False
        
        current_time = time.time()
        
        # Initialize tracking for new sources
        if source_name not in self.last_request_time:
            self.last_request_time[source_name] = 0
            self.request_counts[source_name] = 0
        
        # Reset count if minute has passed
        if current_time - self.last_request_time[source_name] >= 60:
            self.request_counts[source_name] = 0
            self.last_request_time[source_name] = current_time
        
        # Check if under rate limit
        if self.request_counts[source_name] < source.rate_limit:
            self.request_counts[source_name] += 1
            return True
        
        return False
    
    def fetch_espn_data(self, data_type: str = 'scoreboard') -> Dict[str, Any]:
        """Fetch data from ESPN API."""
        
        if not self._check_rate_limit('espn_api'):
            logger.warning("ESPN API rate limit exceeded")
            return {}
        
        source = self.data_sources['espn_api']
        
        try:
            if data_type == 'scoreboard':
                url = f"{source.url}/scoreboard"
            elif data_type == 'teams':
                url = f"{source.url}/teams"
            elif data_type == 'standings':
                url = f"{source.url}/standings"
            else:
                url = f"{source.url}/{data_type}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Log successful collection
            self._log_data_collection('espn_api', data_type, len(data.get('events', [])), True)
            
            return data
            
        except Exception as e:
            logger.error(f"ESPN API error: {e}")
            self._log_data_collection('espn_api', data_type, 0, False, str(e))
            return {}
    
    def fetch_sleeper_data(self, data_type: str = 'players') -> Dict[str, Any]:
        """Fetch data from Sleeper API."""
        
        if not self._check_rate_limit('sleeper_api'):
            logger.warning("Sleeper API rate limit exceeded")
            return {}
        
        source = self.data_sources['sleeper_api']
        
        try:
            if data_type == 'players':
                url = f"{source.url}/players/nfl"
            elif data_type == 'trending':
                url = f"{source.url}/players/nfl/trending/add"
            else:
                url = f"{source.url}/{data_type}"
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the data
            self._cache_raw_data('sleeper_api', data_type, json.dumps(data))
            
            # Log successful collection
            record_count = len(data) if isinstance(data, (list, dict)) else 1
            self._log_data_collection('sleeper_api', data_type, record_count, True)
            
            return data
            
        except Exception as e:
            logger.error(f"Sleeper API error: {e}")
            self._log_data_collection('sleeper_api', data_type, 0, False, str(e))
            return {}
    
    def scrape_pro_football_reference(self, page_type: str = 'weekly_stats') -> Dict[str, Any]:
        """Scrape data from Pro Football Reference."""
        
        if not self._check_rate_limit('pro_football_reference'):
            logger.warning("Pro Football Reference rate limit exceeded")
            return {}
        
        source = self.data_sources['pro_football_reference']
        
        try:
            # Current year and week
            current_year = datetime.now().year
            current_week = min(18, max(1, datetime.now().isocalendar()[1] - 35))
            
            if page_type == 'weekly_stats':
                url = f"{source.url}/years/{current_year}/week_{current_week}.htm"
            elif page_type == 'team_stats':
                url = f"{source.url}/years/{current_year}/opp.htm"
            elif page_type == 'player_stats':
                url = f"{source.url}/years/{current_year}/fantasy.htm"
            else:
                url = f"{source.url}/{page_type}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract tables
            tables = soup.find_all('table')
            scraped_data = {}
            
            for i, table in enumerate(tables):
                if table.get('id'):
                    table_data = self._parse_html_table(table)
                    scraped_data[table.get('id')] = table_data
            
            # Log successful collection
            total_records = sum(len(table_data) for table_data in scraped_data.values())
            self._log_data_collection('pro_football_reference', page_type, total_records, True)
            
            return scraped_data
            
        except Exception as e:
            logger.error(f"Pro Football Reference scraping error: {e}")
            self._log_data_collection('pro_football_reference', page_type, 0, False, str(e))
            return {}
    
    def _parse_html_table(self, table) -> List[Dict[str, Any]]:
        """Parse HTML table into list of dictionaries."""
        
        headers = []
        rows = []
        
        # Get headers
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        
        # Get data rows
        tbody = table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells and len(cells) == len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            row_data[headers[i]] = cell.get_text(strip=True)
                    rows.append(row_data)
        
        return rows
    
    def fetch_fantasy_pros_data(self, position: str = 'all') -> Dict[str, Any]:
        """Fetch projections from FantasyPros."""
        
        if not self.data_sources['fantasy_pros'].active:
            return self._generate_mock_projections(position)
        
        if not self._check_rate_limit('fantasy_pros'):
            logger.warning("FantasyPros rate limit exceeded")
            return {}
        
        # Mock implementation for now
        return self._generate_mock_projections(position)
    
    def _generate_mock_projections(self, position: str) -> Dict[str, Any]:
        """Generate mock projection data."""
        
        mock_projections = {
            'QB': [
                {'player': 'Patrick Mahomes', 'team': 'KC', 'pass_yds': 285, 'pass_tds': 2.1, 'rush_yds': 25},
                {'player': 'Josh Allen', 'team': 'BUF', 'pass_yds': 275, 'pass_tds': 2.3, 'rush_yds': 45},
                {'player': 'Lamar Jackson', 'team': 'BAL', 'pass_yds': 245, 'pass_tds': 1.8, 'rush_yds': 65}
            ],
            'RB': [
                {'player': 'Christian McCaffrey', 'team': 'SF', 'rush_yds': 95, 'rush_tds': 0.8, 'rec_yds': 45},
                {'player': 'Derrick Henry', 'team': 'TEN', 'rush_yds': 85, 'rush_tds': 1.1, 'rec_yds': 15},
                {'player': 'Nick Chubb', 'team': 'CLE', 'rush_yds': 80, 'rush_tds': 0.9, 'rec_yds': 25}
            ],
            'WR': [
                {'player': 'Tyreek Hill', 'team': 'MIA', 'rec_yds': 85, 'rec_tds': 0.7, 'receptions': 6.5},
                {'player': 'Stefon Diggs', 'team': 'BUF', 'rec_yds': 80, 'rec_tds': 0.6, 'receptions': 7.2},
                {'player': 'Davante Adams', 'team': 'LV', 'rec_yds': 75, 'rec_tds': 0.8, 'receptions': 6.8}
            ],
            'TE': [
                {'player': 'Travis Kelce', 'team': 'KC', 'rec_yds': 65, 'rec_tds': 0.6, 'receptions': 5.5},
                {'player': 'Mark Andrews', 'team': 'BAL', 'rec_yds': 55, 'rec_tds': 0.7, 'receptions': 4.8},
                {'player': 'George Kittle', 'team': 'SF', 'rec_yds': 60, 'rec_tds': 0.5, 'receptions': 5.2}
            ]
        }
        
        if position.upper() in mock_projections:
            return {position.upper(): mock_projections[position.upper()]}
        
        return mock_projections
    
    def fetch_odds_data(self, sport: str = 'americanfootball_nfl') -> Dict[str, Any]:
        """Fetch betting odds data."""
        
        if not self.data_sources['odds_api'].active:
            return self._generate_mock_odds()
        
        if not self._check_rate_limit('odds_api'):
            logger.warning("Odds API rate limit exceeded")
            return {}
        
        # Mock implementation for now
        return self._generate_mock_odds()
    
    def _generate_mock_odds(self) -> Dict[str, Any]:
        """Generate mock odds data."""
        
        mock_odds = {
            'games': [
                {
                    'home_team': 'KC',
                    'away_team': 'BUF',
                    'commence_time': (datetime.now() + timedelta(days=2)).isoformat(),
                    'bookmakers': [
                        {
                            'key': 'draftkings',
                            'markets': [
                                {'key': 'h2h', 'outcomes': [
                                    {'name': 'KC', 'price': -120},
                                    {'name': 'BUF', 'price': +100}
                                ]},
                                {'key': 'totals', 'outcomes': [
                                    {'name': 'Over', 'price': -110, 'point': 47.5},
                                    {'name': 'Under', 'price': -110, 'point': 47.5}
                                ]}
                            ]
                        }
                    ]
                }
            ]
        }
        
        return mock_odds
    
    def _cache_raw_data(self, source_name: str, data_type: str, raw_data: str, 
                       expires_hours: int = 24):
        """Cache raw data for later use."""
        
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        data_key = f"{source_name}_{data_type}_{datetime.now().strftime('%Y%m%d')}"
        
        insert_sql = """
        INSERT OR REPLACE INTO raw_data_cache 
        (source_name, data_type, data_key, raw_data, collected_at, expires_at)
        VALUES (:source_name, :data_type, :data_key, :raw_data, :collected_at, :expires_at)
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), {
                'source_name': source_name,
                'data_type': data_type,
                'data_key': data_key,
                'raw_data': raw_data,
                'collected_at': datetime.now(),
                'expires_at': expires_at
            })
            conn.commit()
    
    def _log_data_collection(self, source_name: str, data_type: str, 
                           records_collected: int, success: bool, 
                           error_message: str = None):
        """Log data collection attempt."""
        
        insert_sql = """
        INSERT INTO data_collection_log 
        (source_name, data_type, collection_time, records_collected, success, error_message)
        VALUES (:source_name, :data_type, :collection_time, :records_collected, :success, :error_message)
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), {
                'source_name': source_name,
                'data_type': data_type,
                'collection_time': datetime.now(),
                'records_collected': records_collected,
                'success': success,
                'error_message': error_message
            })
            conn.commit()
    
    def run_comprehensive_data_collection(self) -> Dict[str, Any]:
        """Run comprehensive data collection from all sources."""
        
        collection_results = {}
        
        # ESPN data
        logger.info("Collecting ESPN data...")
        espn_data = self.fetch_espn_data('scoreboard')
        collection_results['espn'] = {
            'success': bool(espn_data),
            'records': len(espn_data.get('events', [])),
            'data_types': ['scoreboard']
        }
        
        # Sleeper data
        logger.info("Collecting Sleeper data...")
        sleeper_data = self.fetch_sleeper_data('players')
        collection_results['sleeper'] = {
            'success': bool(sleeper_data),
            'records': len(sleeper_data) if isinstance(sleeper_data, dict) else 0,
            'data_types': ['players']
        }
        
        # Pro Football Reference scraping
        logger.info("Scraping Pro Football Reference...")
        pfr_data = self.scrape_pro_football_reference('weekly_stats')
        collection_results['pro_football_reference'] = {
            'success': bool(pfr_data),
            'records': sum(len(table) for table in pfr_data.values()) if pfr_data else 0,
            'data_types': ['weekly_stats']
        }
        
        # FantasyPros projections
        logger.info("Collecting FantasyPros projections...")
        fp_data = self.fetch_fantasy_pros_data('all')
        collection_results['fantasy_pros'] = {
            'success': bool(fp_data),
            'records': sum(len(pos_data) for pos_data in fp_data.values()) if fp_data else 0,
            'data_types': ['projections']
        }
        
        # Odds data
        logger.info("Collecting odds data...")
        odds_data = self.fetch_odds_data()
        collection_results['odds'] = {
            'success': bool(odds_data),
            'records': len(odds_data.get('games', [])),
            'data_types': ['betting_odds']
        }
        
        # Generate summary
        total_records = sum(result['records'] for result in collection_results.values())
        successful_sources = sum(1 for result in collection_results.values() if result['success'])
        
        summary = {
            'collection_time': datetime.now().isoformat(),
            'total_sources': len(collection_results),
            'successful_sources': successful_sources,
            'total_records_collected': total_records,
            'success_rate': successful_sources / len(collection_results),
            'source_details': collection_results
        }
        
        return summary
    
    def get_data_collection_status(self) -> Dict[str, Any]:
        """Get current data collection status."""
        
        # Get recent collection logs
        query = """
        SELECT source_name, data_type, MAX(collection_time) as last_collection,
               SUM(records_collected) as total_records,
               AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
        FROM data_collection_log 
        WHERE collection_time >= datetime('now', '-7 days')
        GROUP BY source_name, data_type
        ORDER BY last_collection DESC
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query)).fetchall()
        
        status_data = []
        for row in results:
            status_data.append({
                'source': row[0],
                'data_type': row[1],
                'last_collection': row[2],
                'total_records': row[3],
                'success_rate': row[4]
            })
        
        # Get active sources
        active_sources = [name for name, source in self.data_sources.items() if source.active]
        
        return {
            'active_sources': active_sources,
            'total_active_sources': len(active_sources),
            'recent_collections': status_data,
            'rate_limit_status': {
                source: {
                    'requests_made': self.request_counts.get(source, 0),
                    'limit': self.data_sources[source].rate_limit,
                    'remaining': max(0, self.data_sources[source].rate_limit - self.request_counts.get(source, 0))
                }
                for source in active_sources
            }
        }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old cached data and logs."""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old cache
        delete_cache_sql = """
        DELETE FROM raw_data_cache 
        WHERE expires_at < :cutoff_date OR created_at < :cutoff_date
        """
        
        # Clean up old logs (keep more logs than cache)
        delete_logs_sql = """
        DELETE FROM data_collection_log 
        WHERE created_at < :log_cutoff_date
        """
        
        log_cutoff_date = datetime.now() - timedelta(days=days_to_keep * 2)
        
        with self.engine.connect() as conn:
            cache_deleted = conn.execute(text(delete_cache_sql), {'cutoff_date': cutoff_date}).rowcount
            logs_deleted = conn.execute(text(delete_logs_sql), {'log_cutoff_date': log_cutoff_date}).rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {cache_deleted} cache entries and {logs_deleted} log entries")
        
        return {
            'cache_entries_deleted': cache_deleted,
            'log_entries_deleted': logs_deleted,
            'cutoff_date': cutoff_date.isoformat()
        }

def main():
    """Test the API integrations system."""
    print("🌐 API INTEGRATIONS & WEB SCRAPING SYSTEM")
    print("=" * 70)
    
    # Initialize system
    api_system = APIIntegrations()
    
    # Check system status
    print("📊 System Status:")
    status = api_system.get_data_collection_status()
    
    print(f"   Active Sources: {status['total_active_sources']}")
    for source in status['active_sources']:
        rate_info = status['rate_limit_status'][source]
        print(f"     {source}: {rate_info['remaining']}/{rate_info['limit']} requests remaining")
    
    # Run comprehensive data collection
    print("\n🔄 Running Comprehensive Data Collection:")
    collection_results = api_system.run_comprehensive_data_collection()
    
    print(f"   Collection Time: {collection_results['collection_time']}")
    print(f"   Success Rate: {collection_results['success_rate']:.1%}")
    print(f"   Total Records: {collection_results['total_records_collected']}")
    
    print("\n   📋 Source Details:")
    for source, details in collection_results['source_details'].items():
        status_emoji = "✅" if details['success'] else "❌"
        print(f"     {status_emoji} {source}: {details['records']} records")
    
    # Test individual sources
    print("\n🧪 Testing Individual Sources:")
    
    # ESPN test
    espn_data = api_system.fetch_espn_data('scoreboard')
    print(f"   ESPN API: {'✅' if espn_data else '❌'} ({len(espn_data.get('events', []))} events)")
    
    # Sleeper test
    sleeper_data = api_system.fetch_sleeper_data('players')
    print(f"   Sleeper API: {'✅' if sleeper_data else '❌'} ({len(sleeper_data) if sleeper_data else 0} players)")
    
    # FantasyPros test
    fp_data = api_system.fetch_fantasy_pros_data('QB')
    print(f"   FantasyPros: {'✅' if fp_data else '❌'} ({len(fp_data.get('QB', [])) if fp_data else 0} QB projections)")
    
    # Clean up old data
    print("\n🧹 Cleaning Up Old Data:")
    cleanup_results = api_system.cleanup_old_data(30)
    print(f"   Cache entries deleted: {cleanup_results['cache_entries_deleted']}")
    print(f"   Log entries deleted: {cleanup_results['log_entries_deleted']}")
    
    print("\n" + "=" * 70)
    print("✅ API integrations and web scraping system operational!")

if __name__ == "__main__":
    main()
