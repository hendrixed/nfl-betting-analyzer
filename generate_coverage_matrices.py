#!/usr/bin/env python3
"""
Coverage Matrix Generator
Generates coverage matrices showing which features are available for which statistics
and which models support which markets for validation and reporting.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.database_models import get_db_session, Player, PlayerGameStats, Game
from core.models.feature_engineering import NFLFeatureEngineer

def generate_stats_feature_matrix() -> pd.DataFrame:
    """Generate matrix showing which features are available for which statistics."""
    
    # Define all NFL statistics we track
    stats = [
        'passing_attempts', 'passing_completions', 'passing_yards', 'passing_touchdowns', 'interceptions',
        'rushing_attempts', 'rushing_yards', 'rushing_touchdowns', 'fumbles',
        'targets', 'receptions', 'receiving_yards', 'receiving_touchdowns', 'drops',
        'offensive_snaps', 'snap_percentage', 'routes_run', 'air_yards', 'yac',
        'red_zone_targets', 'goal_line_carries', 'third_down_conversions',
        'field_goals_attempted', 'field_goals_made', 'extra_points_attempted', 'extra_points_made'
    ]
    
    # Define feature categories and their availability
    feature_categories = {
        'rolling_averages_3game': {
            'description': '3-game rolling averages',
            'stats': stats  # All stats have rolling averages
        },
        'rolling_averages_5game': {
            'description': '5-game rolling averages', 
            'stats': stats  # All stats have rolling averages
        },
        'season_to_date': {
            'description': 'Season-to-date totals',
            'stats': stats  # All stats have season totals
        },
        'opponent_adjusted': {
            'description': 'Opponent strength adjustments',
            'stats': ['passing_yards', 'passing_touchdowns', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards', 'receiving_touchdowns']
        },
        'situational_splits': {
            'description': 'Situational performance splits',
            'stats': ['red_zone_targets', 'goal_line_carries', 'third_down_conversions']
        },
        'red_zone_efficiency': {
            'description': 'Red zone efficiency metrics',
            'stats': ['red_zone_targets']
        },
        'goal_line_efficiency': {
            'description': 'Goal line efficiency metrics',
            'stats': ['goal_line_carries']
        },
        'air_yard_distribution': {
            'description': 'Air yards distribution analysis',
            'stats': ['air_yards']
        },
        'yac_efficiency': {
            'description': 'Yards after catch efficiency',
            'stats': ['yac']
        },
        'pressure_splits': {
            'description': 'Performance under pressure',
            'stats': ['passing_yards', 'passing_touchdowns', 'interceptions']
        },
        'coverage_splits': {
            'description': 'Performance vs coverage types',
            'stats': ['targets', 'receptions', 'receiving_yards']
        },
        'rest_days_adjustment': {
            'description': 'Rest days impact adjustment',
            'stats': stats  # All stats affected by rest
        },
        'travel_distance_adjustment': {
            'description': 'Travel distance impact adjustment',
            'stats': stats  # All stats affected by travel
        },
        'weather_adjustment': {
            'description': 'Weather condition adjustments',
            'stats': stats  # All stats affected by weather
        },
        'home_away_splits': {
            'description': 'Home vs away performance',
            'stats': stats  # All stats have home/away splits
        },
        'division_opponent_adjustment': {
            'description': 'Division opponent adjustments',
            'stats': stats  # All stats adjusted for division games
        },
        'pace_adjustment': {
            'description': 'Game pace adjustments',
            'stats': stats  # All stats adjusted for pace
        }
    }
    
    # Create matrix
    matrix_data = []
    for stat in stats:
        row = {'stat': stat}
        for feature_name, feature_info in feature_categories.items():
            row[feature_name] = 1 if stat in feature_info['stats'] else 0
        matrix_data.append(row)
    
    return pd.DataFrame(matrix_data)

def generate_model_market_matrix() -> pd.DataFrame:
    """Generate matrix showing which models support which betting markets."""
    
    # Define betting markets
    markets = [
        'player_passing_yds', 'player_passing_tds', 'player_interceptions',
        'player_rushing_yds', 'player_rushing_tds', 'player_rec_yds', 
        'player_receptions', 'player_rec_tds', 'player_fantasy_points',
        'team_total_points', 'game_total_points', 'point_spread',
        'first_half_total', 'first_td_scorer', 'anytime_td_scorer'
    ]
    
    # Define model types and their market support
    models = {
        'XGBoost': {
            'description': 'Gradient boosting model',
            'markets': ['player_passing_yds', 'player_rushing_yds', 'player_rec_yds', 'player_receptions', 'player_fantasy_points']
        },
        'LightGBM': {
            'description': 'Light gradient boosting',
            'markets': ['player_passing_yds', 'player_rushing_yds', 'player_rec_yds', 'player_receptions', 'player_fantasy_points']
        },
        'RandomForest': {
            'description': 'Random forest ensemble',
            'markets': ['player_passing_yds', 'player_rushing_yds', 'player_rec_yds', 'player_receptions', 'player_fantasy_points']
        },
        'LinearRegression': {
            'description': 'Linear regression baseline',
            'markets': markets  # Linear regression supports all markets
        },
        'NeuralNetwork': {
            'description': 'Deep neural network',
            'markets': ['player_fantasy_points', 'team_total_points', 'game_total_points', 'point_spread']
        },
        'EnsembleModel': {
            'description': 'Ensemble of multiple models',
            'markets': ['player_passing_yds', 'player_rushing_yds', 'player_rec_yds', 'player_receptions', 'player_fantasy_points']
        }
    }
    
    # Create matrix
    matrix_data = []
    for market in markets:
        row = {'market': market}
        for model_name, model_info in models.items():
            row[model_name] = 1 if market in model_info['markets'] else 0
        matrix_data.append(row)
    
    return pd.DataFrame(matrix_data)

def validate_coverage_with_real_data() -> Dict[str, Any]:
    """Validate coverage matrices against actual database data."""
    
    try:
        session = get_db_session("sqlite:///nfl_predictions.db")
        
        # Check available statistics in database
        from sqlalchemy import text
        stats_query = text("""
        SELECT 
            COUNT(CASE WHEN passing_yards IS NOT NULL THEN 1 END) as passing_yards_count,
            COUNT(CASE WHEN rushing_yards IS NOT NULL THEN 1 END) as rushing_yards_count,
            COUNT(CASE WHEN receiving_yards IS NOT NULL THEN 1 END) as receiving_yards_count,
            COUNT(CASE WHEN receptions IS NOT NULL THEN 1 END) as receptions_count,
            COUNT(CASE WHEN fantasy_points_ppr IS NOT NULL THEN 1 END) as fantasy_points_count,
            COUNT(*) as total_records
        FROM player_game_stats
        WHERE created_at >= date('now', '-30 days')
        """)
        
        result = session.execute(stats_query).fetchone()
        
        validation_results = {
            'database_validation': {
                'total_recent_records': result[5] if result else 0,
                'passing_yards_coverage': result[0] if result else 0,
                'rushing_yards_coverage': result[1] if result else 0,
                'receiving_yards_coverage': result[2] if result else 0,
                'receptions_coverage': result[3] if result else 0,
                'fantasy_points_coverage': result[4] if result else 0
            },
            'feature_engineering_validation': {},
            'model_validation': {}
        }
        
        # Test feature engineering with sample data
        try:
            engineer = NFLFeatureEngineer(session)
            
            # Get a sample game for testing
            sample_game = session.query(Game).first()
            if sample_game:
                # Test feature generation
                sample_features = engineer.generate_features(sample_game.game_id, 'QB')
                validation_results['feature_engineering_validation'] = {
                    'features_generated': len(sample_features) if sample_features else 0,
                    'sample_game_id': sample_game.game_id,
                    'feature_categories': list(sample_features.keys()) if sample_features else []
                }
        except Exception as e:
            validation_results['feature_engineering_validation'] = {
                'error': str(e)
            }
        
        # Check model files
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.rglob("*.pkl"))
            validation_results['model_validation'] = {
                'model_files_found': len(model_files),
                'model_files': [str(f.relative_to(models_dir)) for f in model_files]
            }
        
        session.close()
        return validation_results
        
    except Exception as e:
        return {'error': str(e)}

def main():
    """Generate coverage matrices and validation report."""
    print("GENERATING COVERAGE MATRICES")
    print("=" * 50)
    
    # Create reports directory
    reports_dir = Path("reports/coverage")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate stats-feature matrix
    print("Generating stats-feature matrix...")
    stats_matrix = generate_stats_feature_matrix()
    stats_file = reports_dir / "stats_feature_matrix.csv"
    stats_matrix.to_csv(stats_file, index=False)
    print(f"   Saved to {stats_file}")
    
    # Generate model-market matrix
    print("Generating model-market matrix...")
    model_matrix = generate_model_market_matrix()
    model_file = reports_dir / "model_market_matrix.csv"
    model_matrix.to_csv(model_file, index=False)
    print(f"   Saved to {model_file}")
    
    # Validate with real data
    print("Validating coverage with real data...")
    validation_results = validate_coverage_with_real_data()
    validation_file = reports_dir / "coverage_validation.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    print(f"   Validation results saved to {validation_file}")
    
    # Print summary
    print("\nCOVERAGE SUMMARY")
    print("-" * 30)
    print(f"Statistics tracked: {len(stats_matrix)}")
    print(f"Feature categories: {len(stats_matrix.columns) - 1}")
    print(f"Betting markets: {len(model_matrix)}")
    print(f"Model types: {len(model_matrix.columns) - 1}")
    
    if 'database_validation' in validation_results:
        db_val = validation_results['database_validation']
        print(f"Recent database records: {db_val.get('total_recent_records', 0)}")
        print(f"Fantasy points coverage: {db_val.get('fantasy_points_coverage', 0)}")
    
    if 'model_validation' in validation_results:
        model_val = validation_results['model_validation']
        print(f"Model files available: {model_val.get('model_files_found', 0)}")
    
    print("\nCoverage matrices generated successfully!")

if __name__ == "__main__":
    main()
