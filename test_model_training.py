"""
Test script for model training with simplified approach
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from core.database_models import get_db_session, Player, PlayerGameStats
from sqlalchemy import func

def test_simple_model_training():
    """Test model training with actual data"""
    session = get_db_session()
    
    try:
        # Get players with sufficient stats for each position
        for position in ['QB', 'RB', 'WR', 'TE']:
            print(f"\n=== Testing {position} Model ===")
            
            # Get players with at least 5 games
            players_with_stats = session.query(
                Player.player_id,
                Player.name,
                func.count(PlayerGameStats.stat_id).label('game_count'),
                func.avg(PlayerGameStats.fantasy_points_ppr).label('avg_fantasy_points'),
                func.avg(PlayerGameStats.passing_yards).label('avg_passing_yards'),
                func.avg(PlayerGameStats.rushing_yards).label('avg_rushing_yards'),
                func.avg(PlayerGameStats.receiving_yards).label('avg_receiving_yards'),
                func.avg(PlayerGameStats.receptions).label('avg_receptions')
            ).join(PlayerGameStats).filter(
                Player.position == position,
                Player.is_active == True,
                PlayerGameStats.fantasy_points_ppr.isnot(None)
            ).group_by(Player.player_id, Player.name).having(
                func.count(PlayerGameStats.stat_id) >= 5
            ).all()
            
            if len(players_with_stats) < 10:
                print(f"Insufficient data for {position}: {len(players_with_stats)} players")
                continue
            
            # Create feature matrix
            features = []
            targets = []
            
            for player_data in players_with_stats:
                # Simple features based on averages
                feature_row = [
                    player_data.avg_fantasy_points or 0,
                    player_data.avg_passing_yards or 0,
                    player_data.avg_rushing_yards or 0,
                    player_data.avg_receiving_yards or 0,
                    player_data.avg_receptions or 0,
                    player_data.game_count
                ]
                
                features.append(feature_row)
                targets.append(player_data.avg_fantasy_points or 0)
            
            # Convert to arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Remove any rows with NaN or zero targets
            valid_mask = (y > 0) & ~np.isnan(y).any(axis=0)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                print(f"Insufficient valid data for {position}: {len(X)} samples")
                continue
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train simple Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"  Players: {len(players_with_stats)}")
            print(f"  Valid samples: {len(X)}")
            print(f"  Train samples: {len(X_train)}")
            print(f"  Test samples: {len(X_test)}")
            print(f"  R² Score: {r2:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            
            # Feature importance
            feature_names = ['avg_fantasy', 'avg_passing', 'avg_rushing', 'avg_receiving', 'avg_receptions', 'game_count']
            importance = model.feature_importances_
            
            print("  Top features:")
            for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {name}: {imp:.3f}")
    
    finally:
        session.close()

if __name__ == "__main__":
    test_simple_model_training()
