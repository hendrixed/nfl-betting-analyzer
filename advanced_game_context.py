#!/usr/bin/env python3
"""
Advanced Game Context System
Stadium characteristics, rest days, game situation, and environmental factors.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedGameContext:
    """Advanced game context analysis including stadium, weather, and situational factors."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Stadium characteristics database
        self.stadiums = {
            'ARI': {'name': 'State Farm Stadium', 'dome': True, 'elevation': 1117, 'capacity': 63400, 'surface': 'Grass'},
            'ATL': {'name': 'Mercedes-Benz Stadium', 'dome': True, 'elevation': 1050, 'capacity': 71000, 'surface': 'Turf'},
            'BAL': {'name': 'M&T Bank Stadium', 'dome': False, 'elevation': 56, 'capacity': 71008, 'surface': 'Grass'},
            'BUF': {'name': 'Highmark Stadium', 'dome': False, 'elevation': 614, 'capacity': 71608, 'surface': 'Turf'},
            'CAR': {'name': 'Bank of America Stadium', 'dome': False, 'elevation': 748, 'capacity': 75523, 'surface': 'Grass'},
            'CHI': {'name': 'Soldier Field', 'dome': False, 'elevation': 597, 'capacity': 61500, 'surface': 'Grass'},
            'CIN': {'name': 'Paycor Stadium', 'dome': False, 'elevation': 550, 'capacity': 65515, 'surface': 'Turf'},
            'CLE': {'name': 'FirstEnergy Stadium', 'dome': False, 'elevation': 653, 'capacity': 67431, 'surface': 'Grass'},
            'DAL': {'name': 'AT&T Stadium', 'dome': True, 'elevation': 551, 'capacity': 80000, 'surface': 'Turf'},
            'DEN': {'name': 'Empower Field at Mile High', 'dome': False, 'elevation': 5280, 'capacity': 76125, 'surface': 'Grass'},
            'DET': {'name': 'Ford Field', 'dome': True, 'elevation': 585, 'capacity': 65000, 'surface': 'Turf'},
            'GB': {'name': 'Lambeau Field', 'dome': False, 'elevation': 640, 'capacity': 81441, 'surface': 'Grass'},
            'HOU': {'name': 'NRG Stadium', 'dome': True, 'elevation': 43, 'capacity': 72220, 'surface': 'Turf'},
            'IND': {'name': 'Lucas Oil Stadium', 'dome': True, 'elevation': 715, 'capacity': 70000, 'surface': 'Turf'},
            'JAX': {'name': 'TIAA Bank Field', 'dome': False, 'elevation': 16, 'capacity': 67814, 'surface': 'Grass'},
            'KC': {'name': 'Arrowhead Stadium', 'dome': False, 'elevation': 909, 'capacity': 76416, 'surface': 'Grass'},
            'LV': {'name': 'Allegiant Stadium', 'dome': True, 'elevation': 2001, 'capacity': 65000, 'surface': 'Turf'},
            'LAC': {'name': 'SoFi Stadium', 'dome': False, 'elevation': 107, 'capacity': 70240, 'surface': 'Turf'},
            'LAR': {'name': 'SoFi Stadium', 'dome': False, 'elevation': 107, 'capacity': 70240, 'surface': 'Turf'},
            'MIA': {'name': 'Hard Rock Stadium', 'dome': False, 'elevation': 7, 'capacity': 65326, 'surface': 'Grass'},
            'MIN': {'name': 'U.S. Bank Stadium', 'dome': True, 'elevation': 840, 'capacity': 66860, 'surface': 'Turf'},
            'NE': {'name': 'Gillette Stadium', 'dome': False, 'elevation': 131, 'capacity': 66829, 'surface': 'Turf'},
            'NO': {'name': 'Caesars Superdome', 'dome': True, 'elevation': -6, 'capacity': 73208, 'surface': 'Turf'},
            'NYG': {'name': 'MetLife Stadium', 'dome': False, 'elevation': 7, 'capacity': 82500, 'surface': 'Turf'},
            'NYJ': {'name': 'MetLife Stadium', 'dome': False, 'elevation': 7, 'capacity': 82500, 'surface': 'Turf'},
            'PHI': {'name': 'Lincoln Financial Field', 'dome': False, 'elevation': 56, 'capacity': 69596, 'surface': 'Grass'},
            'PIT': {'name': 'Heinz Field', 'dome': False, 'elevation': 745, 'capacity': 68400, 'surface': 'Grass'},
            'SF': {'name': "Levi's Stadium", 'dome': False, 'elevation': 43, 'capacity': 68500, 'surface': 'Grass'},
            'SEA': {'name': 'Lumen Field', 'dome': False, 'elevation': 56, 'capacity': 69000, 'surface': 'Turf'},
            'TB': {'name': 'Raymond James Stadium', 'dome': False, 'elevation': 26, 'capacity': 65890, 'surface': 'Grass'},
            'TEN': {'name': 'Nissan Stadium', 'dome': False, 'elevation': 597, 'capacity': 69143, 'surface': 'Grass'},
            'WAS': {'name': 'FedExField', 'dome': False, 'elevation': 79, 'capacity': 82000, 'surface': 'Grass'}
        }
        
        # Historical weather impact factors
        self.weather_impacts = {
            'wind': {
                'passing': {'low': 1.0, 'medium': 0.95, 'high': 0.85, 'extreme': 0.70},
                'kicking': {'low': 1.0, 'medium': 0.90, 'high': 0.75, 'extreme': 0.60}
            },
            'precipitation': {
                'passing': {'none': 1.0, 'light': 0.98, 'moderate': 0.92, 'heavy': 0.85},
                'rushing': {'none': 1.0, 'light': 1.02, 'moderate': 1.05, 'heavy': 1.08},
                'fumbles': {'none': 1.0, 'light': 1.1, 'moderate': 1.25, 'heavy': 1.4}
            },
            'temperature': {
                'passing': {'hot': 0.95, 'warm': 1.0, 'cool': 1.02, 'cold': 0.98, 'freezing': 0.92},
                'kicking': {'hot': 0.98, 'warm': 1.0, 'cool': 1.0, 'cold': 0.95, 'freezing': 0.85}
            }
        }
        
        # Rest day advantages
        self.rest_advantages = {
            3: 0.95,   # Short rest (Thursday games)
            4: 0.98,   # 4 days rest
            5: 0.99,   # 5 days rest
            6: 1.0,    # Normal rest (Sunday to Sunday)
            7: 1.0,    # Standard week
            8: 1.01,   # Extra day
            9: 1.02,   # 9+ days (bye week advantage)
            10: 1.03,
            14: 1.05   # Coming off bye week
        }
        
        # Travel distance impact (miles)
        self.travel_impact = {
            'short': (0, 500, 1.0),      # No significant impact
            'medium': (500, 1500, 0.98), # Slight negative impact
            'long': (1500, 2500, 0.96),  # Moderate impact
            'coast': (2500, 4000, 0.94)  # Cross-country
        }
        
    def get_stadium_factors(self, team: str) -> Dict[str, Any]:
        """Get stadium-specific factors for a team."""
        stadium_info = self.stadiums.get(team, {})
        
        factors = {
            'dome_advantage': 1.02 if stadium_info.get('dome', False) else 1.0,
            'elevation_factor': self._calculate_elevation_factor(stadium_info.get('elevation', 0)),
            'surface_type': stadium_info.get('surface', 'Grass'),
            'capacity': stadium_info.get('capacity', 70000),
            'crowd_noise_factor': self._calculate_crowd_factor(stadium_info.get('capacity', 70000)),
            'home_field_advantage': self._get_home_field_advantage(team)
        }
        
        return factors
    
    def _calculate_elevation_factor(self, elevation: int) -> float:
        """Calculate performance factor based on elevation."""
        if elevation < 1000:
            return 1.0
        elif elevation < 3000:
            return 1.01  # Slight advantage for kicking
        elif elevation < 5000:
            return 1.03  # Denver effect
        else:
            return 1.05  # Extreme elevation
    
    def _calculate_crowd_factor(self, capacity: int) -> float:
        """Calculate crowd noise impact factor."""
        if capacity < 65000:
            return 0.98
        elif capacity < 75000:
            return 1.0
        else:
            return 1.02  # Large stadium advantage
    
    def _get_home_field_advantage(self, team: str) -> float:
        """Get team-specific home field advantage."""
        # Historical home field advantages (can be updated with real data)
        home_advantages = {
            'SEA': 1.08,  # 12th Man
            'KC': 1.07,   # Arrowhead
            'NO': 1.06,   # Superdome
            'GB': 1.05,   # Lambeau mystique
            'PIT': 1.05,  # Terrible Towel
            'DEN': 1.04,  # Mile High
            'BUF': 1.04,  # Bills Mafia
            'BAL': 1.03,  # Ravens Nest
        }
        
        return home_advantages.get(team, 1.02)  # Default home advantage
    
    def get_weather_impact(self, temperature: float, wind_speed: float, 
                          precipitation: str, dome: bool = False) -> Dict[str, float]:
        """Calculate weather impact on different aspects of the game."""
        
        if dome:
            return {
                'passing_factor': 1.0,
                'rushing_factor': 1.0,
                'kicking_factor': 1.0,
                'fumble_factor': 1.0,
                'total_points_factor': 1.0
            }
        
        # Temperature impact
        if temperature >= 80:
            temp_category = 'hot'
        elif temperature >= 65:
            temp_category = 'warm'
        elif temperature >= 45:
            temp_category = 'cool'
        elif temperature >= 32:
            temp_category = 'cold'
        else:
            temp_category = 'freezing'
        
        # Wind impact
        if wind_speed < 10:
            wind_category = 'low'
        elif wind_speed < 20:
            wind_category = 'medium'
        elif wind_speed < 30:
            wind_category = 'high'
        else:
            wind_category = 'extreme'
        
        # Precipitation impact
        precip_category = precipitation.lower() if precipitation else 'none'
        
        # Calculate factors
        passing_factor = (self.weather_impacts['temperature']['passing'][temp_category] * 
                         self.weather_impacts['wind']['passing'][wind_category] *
                         self.weather_impacts['precipitation']['passing'].get(precip_category, 1.0))
        
        rushing_factor = self.weather_impacts['precipitation']['rushing'].get(precip_category, 1.0)
        
        kicking_factor = (self.weather_impacts['temperature']['kicking'][temp_category] *
                         self.weather_impacts['wind']['kicking'][wind_category])
        
        fumble_factor = self.weather_impacts['precipitation']['fumbles'].get(precip_category, 1.0)
        
        # Overall game impact
        total_points_factor = (passing_factor * 0.6 + rushing_factor * 0.3 + kicking_factor * 0.1)
        
        return {
            'passing_factor': passing_factor,
            'rushing_factor': rushing_factor,
            'kicking_factor': kicking_factor,
            'fumble_factor': fumble_factor,
            'total_points_factor': total_points_factor,
            'weather_severity': self._calculate_weather_severity(temperature, wind_speed, precipitation)
        }
    
    def _calculate_weather_severity(self, temperature: float, wind_speed: float, 
                                  precipitation: str) -> str:
        """Calculate overall weather severity."""
        severity_score = 0
        
        # Temperature severity
        if temperature < 20 or temperature > 95:
            severity_score += 3
        elif temperature < 32 or temperature > 85:
            severity_score += 2
        elif temperature < 40 or temperature > 80:
            severity_score += 1
        
        # Wind severity
        if wind_speed > 25:
            severity_score += 3
        elif wind_speed > 15:
            severity_score += 2
        elif wind_speed > 10:
            severity_score += 1
        
        # Precipitation severity
        if precipitation and precipitation.lower() in ['heavy', 'snow', 'blizzard']:
            severity_score += 3
        elif precipitation and precipitation.lower() in ['moderate', 'rain']:
            severity_score += 2
        elif precipitation and precipitation.lower() in ['light', 'drizzle']:
            severity_score += 1
        
        if severity_score >= 6:
            return 'extreme'
        elif severity_score >= 4:
            return 'severe'
        elif severity_score >= 2:
            return 'moderate'
        else:
            return 'mild'
    
    def get_rest_and_travel_factors(self, team: str, rest_days: int, 
                                  travel_distance: int = 0) -> Dict[str, float]:
        """Calculate rest and travel impact factors."""
        
        # Rest advantage
        rest_factor = self.rest_advantages.get(rest_days, 1.0)
        
        # Travel impact
        travel_factor = 1.0
        for category, (min_dist, max_dist, factor) in self.travel_impact.items():
            if min_dist <= travel_distance < max_dist:
                travel_factor = factor
                break
        
        # Time zone adjustment (simplified)
        timezone_factor = 1.0
        if travel_distance > 1500:  # Cross-country travel
            timezone_factor = 0.97
        elif travel_distance > 1000:  # Significant travel
            timezone_factor = 0.99
        
        return {
            'rest_factor': rest_factor,
            'travel_factor': travel_factor,
            'timezone_factor': timezone_factor,
            'combined_factor': rest_factor * travel_factor * timezone_factor
        }
    
    def get_game_situation_factors(self, score_differential: int, time_remaining: int,
                                 down: int, distance: int, field_position: int) -> Dict[str, float]:
        """Calculate game situation impact factors."""
        
        # Score differential impact (garbage time, blowouts)
        if abs(score_differential) > 21:
            score_factor = 0.8  # Garbage time
        elif abs(score_differential) > 14:
            score_factor = 0.9  # Likely blowout
        elif abs(score_differential) > 7:
            score_factor = 0.95  # Two-score game
        else:
            score_factor = 1.0  # Competitive game
        
        # Time remaining impact
        if time_remaining < 300:  # Less than 5 minutes
            urgency_factor = 1.1
        elif time_remaining < 900:  # Less than 15 minutes
            urgency_factor = 1.05
        else:
            urgency_factor = 1.0
        
        # Down and distance impact
        if down == 4:
            down_factor = 1.2  # Desperation plays
        elif down == 3 and distance > 7:
            down_factor = 1.1  # Passing down
        elif down <= 2 and distance <= 3:
            down_factor = 0.9  # Running down
        else:
            down_factor = 1.0
        
        # Field position impact
        if field_position <= 20:  # Red zone
            red_zone_factor = 1.15
        elif field_position <= 40:  # Plus territory
            red_zone_factor = 1.05
        else:
            red_zone_factor = 1.0
        
        return {
            'score_factor': score_factor,
            'urgency_factor': urgency_factor,
            'down_factor': down_factor,
            'red_zone_factor': red_zone_factor,
            'situation_factor': score_factor * urgency_factor * down_factor * red_zone_factor
        }
    
    def create_game_context_features(self, home_team: str, away_team: str,
                                   temperature: float = 70, wind_speed: float = 5,
                                   precipitation: str = 'none', rest_days_home: int = 7,
                                   rest_days_away: int = 7, travel_distance: int = 0) -> Dict[str, float]:
        """Create comprehensive game context features for ML models."""
        
        # Stadium factors
        home_stadium = self.get_stadium_factors(home_team)
        
        # Weather factors
        weather_impact = self.get_weather_impact(
            temperature, wind_speed, precipitation, home_stadium.get('dome_advantage', 1.0) > 1.0
        )
        
        # Rest and travel factors
        home_rest_travel = self.get_rest_and_travel_factors(home_team, rest_days_home, 0)
        away_rest_travel = self.get_rest_and_travel_factors(away_team, rest_days_away, travel_distance)
        
        features = {
            # Stadium features
            'is_dome': 1.0 if home_stadium.get('dome_advantage', 1.0) > 1.0 else 0.0,
            'elevation_factor': home_stadium['elevation_factor'],
            'is_grass_surface': 1.0 if home_stadium['surface_type'] == 'Grass' else 0.0,
            'stadium_capacity': home_stadium['capacity'] / 100000,  # Normalized
            'home_field_advantage': home_stadium['home_field_advantage'],
            
            # Weather features
            'temperature': temperature / 100,  # Normalized
            'wind_speed': min(wind_speed / 30, 1.0),  # Normalized, capped
            'has_precipitation': 1.0 if precipitation != 'none' else 0.0,
            'weather_severity_score': {'mild': 0.25, 'moderate': 0.5, 'severe': 0.75, 'extreme': 1.0}.get(
                weather_impact['weather_severity'], 0.25
            ),
            
            # Weather impact factors
            'weather_passing_factor': weather_impact['passing_factor'],
            'weather_rushing_factor': weather_impact['rushing_factor'],
            'weather_kicking_factor': weather_impact['kicking_factor'],
            
            # Rest and travel features
            'home_rest_factor': home_rest_travel['rest_factor'],
            'away_rest_factor': away_rest_travel['rest_factor'],
            'travel_factor': away_rest_travel['travel_factor'],
            'timezone_factor': away_rest_travel['timezone_factor'],
            
            # Derived features
            'rest_advantage': home_rest_travel['rest_factor'] - away_rest_travel['rest_factor'],
            'total_game_factor': (weather_impact['total_points_factor'] * 
                                home_rest_travel['combined_factor'] * 
                                away_rest_travel['combined_factor'])
        }
        
        return features

def main():
    """Test the advanced game context system."""
    print("🏟️ Advanced Game Context Analysis System")
    print("=" * 60)
    
    # Initialize system
    context = AdvancedGameContext()
    
    # Test stadium factors
    print("🏈 Stadium Analysis (Arrowhead Stadium - KC):")
    kc_stadium = context.get_stadium_factors('KC')
    print(f"   Dome: {'Yes' if kc_stadium.get('dome_advantage', 1.0) > 1.0 else 'No'}")
    print(f"   Elevation Factor: {kc_stadium['elevation_factor']:.3f}")
    print(f"   Surface: {kc_stadium['surface_type']}")
    print(f"   Capacity: {kc_stadium['capacity']:,}")
    print(f"   Home Field Advantage: {kc_stadium['home_field_advantage']:.3f}")
    
    # Test weather impact
    print("\n🌦️ Weather Impact Analysis (Cold, Windy Game):")
    weather = context.get_weather_impact(
        temperature=25, wind_speed=20, precipitation='light', dome=False
    )
    print(f"   Passing Factor: {weather['passing_factor']:.3f}")
    print(f"   Rushing Factor: {weather['rushing_factor']:.3f}")
    print(f"   Kicking Factor: {weather['kicking_factor']:.3f}")
    print(f"   Weather Severity: {weather['weather_severity']}")
    
    # Test rest and travel
    print("\n✈️ Rest and Travel Analysis:")
    rest_travel = context.get_rest_and_travel_factors('LAR', rest_days=6, travel_distance=2500)
    print(f"   Rest Factor: {rest_travel['rest_factor']:.3f}")
    print(f"   Travel Factor: {rest_travel['travel_factor']:.3f}")
    print(f"   Timezone Factor: {rest_travel['timezone_factor']:.3f}")
    print(f"   Combined Factor: {rest_travel['combined_factor']:.3f}")
    
    # Test game situation
    print("\n⏰ Game Situation Analysis (Close 4th Quarter):")
    situation = context.get_game_situation_factors(
        score_differential=3, time_remaining=180, down=3, distance=8, field_position=25
    )
    print(f"   Score Factor: {situation['score_factor']:.3f}")
    print(f"   Urgency Factor: {situation['urgency_factor']:.3f}")
    print(f"   Down Factor: {situation['down_factor']:.3f}")
    print(f"   Red Zone Factor: {situation['red_zone_factor']:.3f}")
    
    # Test comprehensive features
    print("\n🔧 Comprehensive Game Context Features (KC vs BUF):")
    features = context.create_game_context_features(
        home_team='KC', away_team='BUF',
        temperature=35, wind_speed=15, precipitation='none',
        rest_days_home=7, rest_days_away=6, travel_distance=1200
    )
    
    print("   Key Features:")
    key_features = ['is_dome', 'home_field_advantage', 'weather_passing_factor', 
                   'rest_advantage', 'total_game_factor']
    for feature in key_features:
        print(f"   {feature}: {features[feature]:.3f}")
    
    print(f"\n   Total Features Generated: {len(features)}")
    print("\n✅ Advanced game context system test complete!")

if __name__ == "__main__":
    main()
