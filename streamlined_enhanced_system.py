#!/usr/bin/env python3
"""
Streamlined Enhanced NFL Betting System
Optimized for performance with existing data structure.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlinedEnhancedPredictor:
    """Streamlined enhanced predictor that works efficiently with existing data."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        self.models = {}
        self.model_dir = Path("models/streamlined")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Prediction targets optimized for existing data
        self.prediction_targets = {
            'QB': ['passing_yards', 'passing_touchdowns', 'passing_attempts', 'fantasy_points_ppr'],
            'RB': ['rushing_yards', 'rushing_touchdowns', 'rushing_attempts', 'fantasy_points_ppr'],
            'WR': ['receiving_yards', 'receiving_touchdowns', 'receptions', 'targets', 'fantasy_points_ppr'],
            'TE': ['receiving_yards', 'receiving_touchdowns', 'receptions', 'targets', 'fantasy_points_ppr']
        }
        
    def get_streamlined_features(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """Generate streamlined features without complex historical lookups."""
        features = df.copy()
        
        # Basic statistical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove target columns and identifiers
        exclude_cols = ['stat_id', 'player_id', 'game_id', 'created_at', 'updated_at']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create interaction features for the position
        if position == 'QB':
            if 'passing_attempts' in features.columns and 'passing_completions' in features.columns:
                features['completion_rate'] = features['passing_completions'] / (features['passing_attempts'] + 1)
            if 'passing_yards' in features.columns and 'passing_attempts' in features.columns:
                features['yards_per_attempt'] = features['passing_yards'] / (features['passing_attempts'] + 1)
                
        elif position in ['RB']:
            if 'rushing_yards' in features.columns and 'rushing_attempts' in features.columns:
                features['yards_per_carry'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
                
        elif position in ['WR', 'TE']:
            if 'receiving_yards' in features.columns and 'receptions' in features.columns:
                features['yards_per_reception'] = features['receiving_yards'] / (features['receptions'] + 1)
            if 'receptions' in features.columns and 'targets' in features.columns:
                features['catch_rate'] = features['receptions'] / (features['targets'] + 1)
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Select only numeric feature columns
        feature_cols = [col for col in features.columns if col in feature_cols or col.endswith('_rate') or col.endswith('_per_')]
        
        return features[feature_cols]
    
    def prepare_training_data(self, position: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for a specific position and target."""
        
        # Get data for the position
        query = text(f"""
            SELECT *
            FROM player_game_stats
            WHERE player_id LIKE '%_{position.lower()}'
            AND {target} IS NOT NULL
            AND {target} >= 0
            ORDER BY created_at DESC
            LIMIT 2000
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if len(df) < 10:
            logger.warning(f"Insufficient data for {position} {target}: {len(df)} samples")
            return pd.DataFrame(), pd.Series()
        
        # Prepare features
        X = self.get_streamlined_features(df, position)
        y = df[target]
        
        # Remove any remaining NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features for {position} {target}")
        return X, y
    
    def train_model(self, position: str, target: str) -> Optional[Dict[str, Any]]:
        """Train a model for specific position and target."""
        
        X, y = self.prepare_training_data(position, target)
        
        if len(X) < 10:
            return None
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=10,
                random_state=42,
                n_jobs=1  # Single thread to avoid issues
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            model_name = f"{position}_{target}_streamlined"
            model_path = self.model_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            
            # Store in memory
            self.models[model_name] = model
            
            logger.info(f"✅ {model_name}: R² = {r2:.3f}, MSE = {mse:.2f}")
            
            return {
                'model_name': model_name,
                'r2_score': r2,
                'mse': mse,
                'n_samples': len(X),
                'n_features': len(X.columns)
            }
            
        except Exception as e:
            logger.error(f"❌ Error training {position} {target}: {e}")
            return None
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all models efficiently."""
        
        results = {
            'trained_models': [],
            'failed_models': [],
            'summary': {}
        }
        
        total_models = sum(len(targets) for targets in self.prediction_targets.values())
        trained_count = 0
        
        print(f"🤖 Training {total_models} streamlined models...")
        
        for position, targets in self.prediction_targets.items():
            print(f"\n📊 Training {position} models...")
            
            position_results = []
            
            for target in targets:
                print(f"  🎯 {target}...", end=" ")
                
                result = self.train_model(position, target)
                
                if result:
                    results['trained_models'].append(result)
                    position_results.append(result)
                    trained_count += 1
                    print("✅")
                else:
                    results['failed_models'].append(f"{position}_{target}")
                    print("❌")
            
            if position_results:
                avg_r2 = np.mean([r['r2_score'] for r in position_results])
                results['summary'][position] = {
                    'models_trained': len(position_results),
                    'avg_r2': avg_r2
                }
        
        results['summary']['total'] = {
            'trained': trained_count,
            'failed': len(results['failed_models']),
            'success_rate': trained_count / total_models if total_models > 0 else 0
        }
        
        return results
    
    def load_models(self) -> int:
        """Load existing models with compatibility checking."""
        loaded = 0
        
        for model_file in self.model_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                model = joblib.load(model_file)
                
                # Check if model is compatible (RandomForestRegressor)
                if hasattr(model, 'n_estimators') and hasattr(model, 'predict'):
                    self.models[model_name] = model
                    loaded += 1
                else:
                    logger.warning(f"Incompatible model type in {model_file}, removing...")
                    model_file.unlink()  # Delete incompatible model
                    
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
                # Remove corrupted model file
                try:
                    model_file.unlink()
                except:
                    pass
        
        return loaded
    
    def predict(self, player_id: str, target: str) -> Optional[Dict[str, Any]]:
        """Make a prediction for a player and target."""
        
        # Extract position from player_id
        position = None
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if player_id.endswith(f'_{pos.lower()}'):
                position = pos
                break
        
        if not position:
            return None
        
        model_name = f"{position}_{target}_streamlined"
        
        if model_name not in self.models:
            return None
        
        try:
            # Get recent data for the player
            query = text(f"""
                SELECT *
                FROM player_game_stats
                WHERE player_id = :player_id
                ORDER BY created_at DESC
                LIMIT 5
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'player_id': player_id})
            
            if len(df) == 0:
                return None
            
            # Use most recent game for features
            recent_game = df.iloc[0:1]
            X = self.get_streamlined_features(recent_game, position)
            
            if len(X.columns) == 0:
                return None
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X)[0]
            
            # Calculate confidence based on recent performance
            if len(df) >= 3:
                recent_values = df[target].head(3).dropna()
                if len(recent_values) > 0:
                    std_dev = recent_values.std()
                    confidence = max(0.1, min(0.9, 1 / (1 + std_dev / (recent_values.mean() + 1))))
                else:
                    confidence = 0.5
            else:
                confidence = 0.3
            
            return {
                'player_id': player_id,
                'target': target,
                'prediction': float(prediction),
                'confidence': float(confidence),
                'recent_avg': float(df[target].head(3).mean()) if len(df) >= 3 else None
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {player_id} {target}: {e}")
            return None
    
    def generate_recommendations(self, min_confidence: float = 0.4) -> List[Dict[str, Any]]:
        """Generate betting recommendations."""
        
        recommendations = []
        
        # Get active players (those with recent games)
        query = text("""
            SELECT DISTINCT player_id,
                   COUNT(*) as game_count,
                   MAX(created_at) as last_game
            FROM player_game_stats
            WHERE created_at >= date('now', '-30 days')
            GROUP BY player_id
            HAVING game_count >= 2
            ORDER BY last_game DESC
            LIMIT 50
        """)
        
        with self.engine.connect() as conn:
            active_players = pd.read_sql(query, conn)
        
        print(f"🎯 Generating recommendations for {len(active_players)} active players...")
        
        for _, player_row in active_players.iterrows():
            player_id = player_row['player_id']
            
            # Extract position
            position = None
            for pos in ['QB', 'RB', 'WR', 'TE']:
                if player_id.endswith(f'_{pos.lower()}'):
                    position = pos
                    break
            
            if not position or position not in self.prediction_targets:
                continue
            
            # Generate predictions for each target
            for target in self.prediction_targets[position]:
                prediction = self.predict(player_id, target)
                
                if prediction and prediction['confidence'] >= min_confidence:
                    # Add recommendation logic
                    pred_value = prediction['prediction']
                    recent_avg = prediction.get('recent_avg', pred_value)
                    
                    # Simple over/under recommendation
                    if pred_value > recent_avg * 1.1:
                        recommendation = "OVER"
                        edge = (pred_value - recent_avg) / recent_avg
                    elif pred_value < recent_avg * 0.9:
                        recommendation = "UNDER"
                        edge = (recent_avg - pred_value) / recent_avg
                    else:
                        continue  # No strong recommendation
                    
                    recommendations.append({
                        'player_id': player_id,
                        'position': position,
                        'target': target,
                        'prediction': pred_value,
                        'recent_avg': recent_avg,
                        'recommendation': recommendation,
                        'confidence': prediction['confidence'],
                        'edge': edge,
                        'last_game': player_row['last_game']
                    })
        
        # Sort by confidence and edge
        recommendations.sort(key=lambda x: (x['confidence'] * x['edge']), reverse=True)
        
        return recommendations[:20]  # Top 20 recommendations

def main():
    """Main function to run the streamlined enhanced system."""
    
    print("🏈 STREAMLINED ENHANCED NFL BETTING ANALYZER")
    print("=" * 60)
    print("🚀 Optimized for performance with existing data structure")
    print()
    
    try:
        # Initialize predictor
        predictor = StreamlinedEnhancedPredictor()
        
        # Load existing models
        loaded_models = predictor.load_models()
        print(f"📂 Loaded {loaded_models} existing models")
        
        # Train new models if needed
        if loaded_models < 10:
            print("\n🤖 Training streamlined models...")
            results = predictor.train_all_models()
            
            print(f"\n📊 TRAINING RESULTS:")
            print(f"✅ Successfully trained: {results['summary']['total']['trained']}")
            print(f"❌ Failed to train: {results['summary']['total']['failed']}")
            print(f"📈 Success rate: {results['summary']['total']['success_rate']:.1%}")
            
            if results['summary']:
                print(f"\n📋 Position Summary:")
                for position, stats in results['summary'].items():
                    if position != 'total':
                        print(f"  {position}: {stats['models_trained']} models, avg R² = {stats['avg_r2']:.3f}")
        
        # Generate recommendations
        print(f"\n🎯 Generating betting recommendations...")
        recommendations = predictor.generate_recommendations()
        
        if recommendations:
            print(f"\n💡 TOP BETTING RECOMMENDATIONS:")
            print("-" * 80)
            
            for i, rec in enumerate(recommendations[:10], 1):
                player_name = rec['player_id'].replace('_', ' ').title()
                print(f"{i:2d}. {player_name} ({rec['position']})")
                print(f"    📊 {rec['target']}: {rec['recommendation']} {rec['prediction']:.1f}")
                print(f"    🎯 Confidence: {rec['confidence']:.1%} | Edge: {rec['edge']:.1%}")
                print(f"    📈 Recent Avg: {rec['recent_avg']:.1f}")
                print()
        else:
            print("❌ No recommendations available")
        
        print("✅ Streamlined enhanced system completed successfully!")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
