#!/usr/bin/env python3
"""
NFL Prediction System - API Setup and Testing Script
Helps you set up and test all required APIs.
"""

import os
import requests
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import nfl_data_py as nfl

class APISetupHelper:
    """Helper class to setup and test APIs."""
    
    def __init__(self):
        self.results = {}
        
    def test_all_apis(self) -> Dict[str, bool]:
        """Test all APIs and return status."""
        print("🏈 NFL Prediction System - API Setup & Testing")
        print("=" * 50)
        
        tests = [
            ("NFL Data (nfl_data_py)", self._test_nfl_data_py),
            ("ESPN API (Free)", self._test_espn_api),
            ("The Odds API", self._test_odds_api),
            ("OpenWeatherMap API", self._test_weather_api),
            ("NewsAPI", self._test_news_api)
        ]
        
        for name, test_func in tests:
            print(f"\n🔍 Testing {name}...")
            try:
                success, message = test_func()
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   {status}: {message}")
                self.results[name] = success
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                self.results[name] = False
                
        self._print_summary()
        return self.results
        
    def _test_nfl_data_py(self) -> Tuple[bool, str]:
        """Test nfl_data_py package."""
        try:
            # Test basic import
            import nfl_data_py as nfl
            
            # Test data retrieval (small sample)
            rosters = nfl.import_rosters([2024], columns=['player_id', 'player_name', 'position'])
            
            if len(rosters) > 0:
                return True, f"Successfully loaded {len(rosters)} player records"
            else:
                return False, "No data returned"
                
        except ImportError:
            return False, "nfl_data_py not installed. Run: pip install nfl-data-py"
        except Exception as e:
            return False, f"Error: {str(e)}"
            
    def _test_espn_api(self) -> Tuple[bool, str]:
        """Test ESPN API (no key required)."""
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                return True, f"Successfully retrieved {len(events)} games"
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
            
    def _test_odds_api(self) -> Tuple[bool, str]:
        """Test The Odds API."""
        api_key = os.getenv('ODDS_API_KEY')
        
        if not api_key:
            return False, "No API key found. Set ODDS_API_KEY environment variable"
            
        try:
            url = "https://api.the-odds-api.com/v4/sports"
            params = {'api_key': api_key}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                sports = response.json()
                # Check remaining requests
                remaining = response.headers.get('x-requests-remaining', 'unknown')
                return True, f"API working. {remaining} requests remaining"
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
            
    def _test_weather_api(self) -> Tuple[bool, str]:
        """Test OpenWeatherMap API."""
        api_key = os.getenv('WEATHER_API_KEY')
        
        if not api_key:
            return False, "No API key found. Set WEATHER_API_KEY environment variable"
            
        try:
            # Test with Kansas City (Arrowhead Stadium)
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': 39.0489,
                'lon': -94.4839, 
                'appid': api_key,
                'units': 'imperial'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                return True, f"Weather API working. Current temp: {temp}°F"
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
            
    def _test_news_api(self) -> Tuple[bool, str]:
        """Test NewsAPI."""
        api_key = os.getenv('NEWS_API_KEY')
        
        if not api_key:
            return False, "No API key found. Set NEWS_API_KEY environment variable"
            
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': api_key,
                'q': 'NFL',
                'pageSize': 1
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('totalResults', 0)
                return True, f"NewsAPI working. {total} articles available"
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
            
    def _print_summary(self):
        """Print test summary and recommendations."""
        print("\n" + "=" * 50)
        print("📊 API TEST SUMMARY")
        print("=" * 50)
        
        working_count = sum(1 for status in self.results.values() if status)
        total_count = len(self.results)
        
        print(f"\n✅ Working APIs: {working_count}/{total_count}")
        
        if self.results.get("NFL Data (nfl_data_py)", False):
            print("\n🎉 GOOD NEWS: You can start with FREE data!")
            print("   - Core NFL stats, schedules, rosters available")
            print("   - No API keys required")
            
        print(f"\n📋 RECOMMENDATIONS:")
        
        if not self.results.get("The Odds API", False):
            print("   🎯 Get betting data:")
            print("     1. Visit: https://the-odds-api.com/")
            print("     2. Sign up for free account (500 requests/month)")
            print("     3. Set ODDS_API_KEY environment variable")
            
        if not self.results.get("OpenWeatherMap API", False):
            print("   🌤️  Get weather data:")
            print("     1. Visit: https://openweathermap.org/api")
            print("     2. Sign up for free account (1,000 calls/day)")
            print("     3. Set WEATHER_API_KEY environment variable")
            
        if not self.results.get("NewsAPI", False):
            print("   📰 Get news data (optional):")
            print("     1. Visit: https://newsapi.org/")
            print("     2. Sign up for free account (1,000 requests/month)")
            print("     3. Set NEWS_API_KEY environment variable")
            
        print(f"\n🚀 NEXT STEPS:")
        if working_count >= 2:  # NFL data + ESPN at minimum
            print("   ✅ You can start using the system now!")
            print("   ✅ Run: python complete_demo.py")
        else:
            print("   ⚠️  Setup at least the NFL data first")
            print("   ⚠️  Run: pip install nfl-data-py")


def create_env_template():
    """Create .env template file."""
    env_template = """# NFL Prediction System - Environment Variables
# Copy this to .env and fill in your API keys

# Database
DATABASE_URL=sqlite:///nfl_predictions.db

# The Odds API (Betting Data)
# Get free key at: https://the-odds-api.com/
ODDS_API_KEY=your_odds_api_key_here

# OpenWeatherMap API (Weather Data) 
# Get free key at: https://openweathermap.org/api
WEATHER_API_KEY=your_weather_api_key_here

# NewsAPI (News & Injuries)
# Get free key at: https://newsapi.org/
NEWS_API_KEY=your_news_api_key_here

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env.template")
    with open(env_file, 'w') as f:
        f.write(env_template.strip())
        
    print(f"📄 Created .env.template file")
    print("   Copy to .env and add your API keys")


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing required dependencies...")
    
    dependencies = [
        "nfl-data-py",
        "requests", 
        "python-dotenv",
        "pandas",
        "numpy"
    ]
    
    import subprocess
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {dep}")


def main():
    """Main setup function."""
    print("🏈 NFL Prediction System - API Setup")
    print("=" * 40)
    
    # Install dependencies
    print("\n1️⃣ Installing dependencies...")
    install_dependencies()
    
    # Create environment template
    print("\n2️⃣ Creating environment template...")
    create_env_template()
    
    # Load environment variables
    print("\n3️⃣ Loading environment variables...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("   ✅ Environment loaded")
    except ImportError:
        print("   ⚠️  python-dotenv not available")
    
    # Test APIs
    print("\n4️⃣ Testing APIs...")
    helper = APISetupHelper()
    results = helper.test_all_apis()
    
    # Final instructions
    if results.get("NFL Data (nfl_data_py)", False):
        print(f"\n🎉 Setup complete! You can now:")
        print(f"   python complete_demo.py")
    else:
        print(f"\n⚠️  Please install nfl-data-py first:")
        print(f"   pip install nfl-data-py")


if __name__ == "__main__":
    main()