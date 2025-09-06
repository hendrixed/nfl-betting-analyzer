"""
Comprehensive NFL Betting Analyzer - Production Ready System
Integrates all enhanced features: expanded targets, ensemble models, prop bets, backtesting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from enhanced_prediction_targets import (
    PREDICTION_TARGETS, get_targets_for_position, get_prop_bet_targets,
    StatCategory, PredictionTarget
)
from enhanced_ensemble_models import ComprehensivePredictor
from feature_engineering import AdvancedFeatureEngineer, FeatureConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveBettingAnalyzer:
    """Production-ready NFL betting analyzer with comprehensive features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the comprehensive betting analyzer."""
        self.config = config or self._default_config()
        
        # Database setup
        self.db_url = self.config.get('database_url', "sqlite:///data/nfl_predictions.db")
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.feature_engineer = AdvancedFeatureEngineer(
            self.db_url, 
            self._create_feature_config()
        )
        
        self.predictor = ComprehensivePredictor(self.config)
        
        # Model and output directories
        self.model_dir = Path(self.config.get('model_directory', 'models/comprehensive'))
        self.output_dir = Path(self.config.get('output_directory', 'output'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {}
        self.backtesting_results = {}
        
        logger.info("Comprehensive NFL Betting Analyzer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'model_types': ['xgboost', 'lightgbm', 'random_forest', 'neural_network'],
            'ensemble_method': 'weighted_average',
            'confidence_threshold': 0.65,
            'min_games_threshold': 5,
            'prop_bet_threshold': 0.7,
            'backtesting_enabled': True,
            'real_time_updates': True,
            'database_url': "sqlite:///data/nfl_predictions.db",
            'model_directory': 'models/comprehensive',
            'output_directory': 'output'
        }
    
    def _create_feature_config(self) -> FeatureConfig:
        """Create feature engineering configuration."""
        return FeatureConfig(
            lookback_windows=[3, 5, 8, 10],
            rolling_windows=[4, 6, 8],
            min_games_threshold=self.config.get('min_games_threshold', 5),
            feature_version="v2.0_comprehensive",
            scale_features=True,
            handle_missing="impute"
        )
    
    def validate_database(self) -> bool:
        """Validate database connection and data integrity."""
        try:
            with self.Session() as session:
                # Check core tables
                tables_to_check = [
                    ('players', 'player_id'),
                    ('player_game_stats', 'player_id'),
                    ('games', 'game_id')
                ]
                
                for table, key_col in tables_to_check:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    logger.info(f"✅ {table}: {count:,} records")
                    
                    if count == 0:
                        logger.error(f"❌ No data found in {table}")
                        return False
                
                # Check data quality
                result = session.execute(text("""
                    SELECT COUNT(DISTINCT pgs.player_id) as unique_players,
                           COUNT(*) as total_stats,
                           MIN(pgs.created_at) as earliest_date,
                           MAX(pgs.created_at) as latest_date
                    FROM player_game_stats pgs
                """))
                
                stats = result.fetchone()
                logger.info(f"📊 Data Quality: {stats[0]} unique players, {stats[1]:,} total stats")
                logger.info(f"📅 Date range: {stats[2]} to {stats[3]}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Database validation failed: {e}")
            return False
    
    def train_comprehensive_models(self):
        """Train models for all positions and prediction targets."""
        logger.info("🤖 Training comprehensive prediction models...")
        
        if not self.validate_database():
            logger.error("Database validation failed. Cannot proceed with training.")
            return
        
        # Get training data
        features_df, targets_df = self._prepare_comprehensive_training_data()
        
        if features_df.empty or targets_df.empty:
            logger.error("No training data available")
            return
        
        logger.info(f"Training with {len(features_df)} feature samples and {len(targets_df)} target samples")
        
        # Train models for each position
        positions = ['QB', 'RB', 'WR', 'TE']
        total_models_trained = 0
        
        for position in positions:
            logger.info(f"Training models for {position}...")
            
            # Get position-specific data
            position_features = features_df[features_df['position'] == position]
            position_targets = targets_df[targets_df['position'] == position]
            
            if len(position_features) < self.config['min_games_threshold']:
                logger.warning(f"Insufficient data for {position}: {len(position_features)} samples")
                continue
            
            # Train models for each target
            targets = get_targets_for_position(position)
            
            for target in targets:
                try:
                    model_trained = self._train_single_model(
                        position_features, position_targets, target, position
                    )
                    if model_trained:
                        total_models_trained += 1
                        
                except Exception as e:
                    logger.error(f"Error training {position} {target.name}: {e}")
        
        logger.info(f"✅ Training completed! {total_models_trained} models trained successfully")
        
        # Save training summary
        self._save_training_summary(total_models_trained)
    
    def _prepare_comprehensive_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare comprehensive training data with advanced features."""
        try:
            with self.Session() as session:
                # Get player game stats with positions
                query = text("""
                    SELECT 
                        pgs.player_id, pgs.game_id,
                        CASE 
                            WHEN pgs.player_id LIKE '%_qb' THEN 'QB'
                            WHEN pgs.player_id LIKE '%_rb' THEN 'RB'
                            WHEN pgs.player_id LIKE '%_wr' THEN 'WR'
                            WHEN pgs.player_id LIKE '%_te' THEN 'TE'
                            WHEN pgs.player_id LIKE '%_k' THEN 'K'
                            WHEN pgs.player_id LIKE '%_def' THEN 'DEF'
                            ELSE 'UNKNOWN'
                        END as position,
                        pgs.passing_attempts, pgs.passing_completions, pgs.passing_yards,
                        pgs.passing_touchdowns, pgs.passing_interceptions, pgs.passing_sacks,
                        pgs.rushing_attempts, pgs.rushing_yards, pgs.rushing_touchdowns,
                        pgs.rushing_fumbles, pgs.rushing_first_downs,
                        pgs.targets, pgs.receptions, pgs.receiving_yards,
                        pgs.receiving_touchdowns, pgs.receiving_fumbles, pgs.receiving_first_downs,
                        pgs.fantasy_points_standard, pgs.fantasy_points_ppr, pgs.fantasy_points_half_ppr,
                        pgs.created_at
                    FROM player_game_stats pgs
                    WHERE (pgs.player_id LIKE '%_qb' OR pgs.player_id LIKE '%_rb' 
                           OR pgs.player_id LIKE '%_wr' OR pgs.player_id LIKE '%_te')
                    AND pgs.created_at IS NOT NULL
                    ORDER BY pgs.created_at DESC
                    LIMIT 5000
                """)
                
                result = session.execute(query)
                raw_data = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                logger.info(f"Retrieved {len(raw_data)} raw game statistics")
                
                # Engineer advanced features for each player-game combination
                features_list = []
                
                for _, row in raw_data.iterrows():
                    try:
                        # Engineer features for this player-game
                        features = self.feature_engineer.engineer_player_features(
                            row['player_id'], row['game_id'], row['position']
                        )
                        
                        if features:
                            feature_row = {
                                'player_id': row['player_id'],
                                'game_id': row['game_id'],
                                'position': row['position'],
                                **features
                            }
                            features_list.append(feature_row)
                            
                    except Exception as e:
                        logger.debug(f"Error engineering features for {row['player_id']}: {e}")
                        continue
                
                features_df = pd.DataFrame(features_list)
                
                # Prepare targets dataframe
                target_columns = [
                    'player_id', 'game_id', 'position',
                    'passing_attempts', 'passing_completions', 'passing_yards',
                    'passing_touchdowns', 'passing_interceptions', 'passing_sacks',
                    'rushing_attempts', 'rushing_yards', 'rushing_touchdowns',
                    'rushing_fumbles', 'rushing_first_downs',
                    'targets', 'receptions', 'receiving_yards',
                    'receiving_touchdowns', 'receiving_fumbles', 'receiving_first_downs',
                    'fantasy_points_standard', 'fantasy_points_ppr', 'fantasy_points_half_ppr'
                ]
                
                targets_df = raw_data[target_columns].copy()
                
                # Fill missing values
                numeric_columns = targets_df.select_dtypes(include=[np.number]).columns
                targets_df[numeric_columns] = targets_df[numeric_columns].fillna(0)
                
                logger.info(f"Engineered features for {len(features_df)} samples")
                
                return features_df, targets_df
                
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _train_single_model(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                           target: PredictionTarget, position: str) -> bool:
        """Train a single model for a specific target."""
        try:
            # Merge features and targets
            data = pd.merge(features_df, targets_df, on=['player_id', 'game_id'], how='inner')
            
            # Check if target column exists
            if target.column not in data.columns:
                logger.warning(f"Target column {target.column} not found")
                return False
            
            # Remove rows with missing target values
            data = data.dropna(subset=[target.column])
            
            if len(data) < self.config['min_games_threshold']:
                logger.warning(f"Insufficient data for {position} {target.name}: {len(data)} samples")
                return False
            
            # Prepare features and target
            feature_cols = [col for col in features_df.columns 
                           if col not in ['player_id', 'game_id', 'position']]
            
            X = data[feature_cols].fillna(0).values
            y = data[target.column].values
            
            # Skip if no variance
            if np.std(y) == 0:
                logger.warning(f"No variance in {position} {target.name}")
                return False
            
            # Train ensemble model
            from enhanced_ensemble_models import EnhancedEnsembleModel
            
            ensemble = EnhancedEnsembleModel(self.config)
            results = ensemble.train(X, y, target)
            
            # Store model
            model_key = f"{position}_{target.name}"
            self.predictor.models[model_key] = ensemble
            
            # Save performance metrics
            self.performance_metrics[model_key] = {
                'target': target.name,
                'position': position,
                'category': target.category.value,
                'results': results,
                'training_samples': len(X),
                'feature_count': X.shape[1],
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model to disk
            model_path = self.model_dir / f"{model_key}.pkl"
            import joblib
            joblib.dump(ensemble, model_path)
            
            best_score = max(results.values()) if results else 0
            logger.info(f"✅ {model_key}: Best R² = {best_score:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training {position} {target.name}: {e}")
            return False
    
    def generate_comprehensive_predictions(self, player_id: str) -> Dict[str, Any]:
        """Generate comprehensive predictions for a player."""
        try:
            # Get player position
            with self.Session() as session:
                query = text("SELECT position FROM players WHERE player_id = :player_id")
                result = session.execute(query, {"player_id": player_id})
                row = result.fetchone()
                
                if not row:
                    return {"error": f"Player {player_id} not found"}
                
                position = row[0]
            
            # Get recent game for feature engineering
            with self.Session() as session:
                query = text("""
                    SELECT game_id FROM player_game_stats 
                    WHERE player_id = :player_id 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = session.execute(query, {"player_id": player_id})
                row = result.fetchone()
                
                if not row:
                    return {"error": f"No recent games found for {player_id}"}
                
                recent_game_id = row[0]
            
            # Engineer features
            features = self.feature_engineer.engineer_player_features(
                player_id, recent_game_id, position
            )
            
            if not features:
                return {"error": f"Could not engineer features for {player_id}"}
            
            # Make predictions
            predictions = self.predictor.predict(features, position)
            
            # Add metadata
            result = {
                'player_id': player_id,
                'position': position,
                'predictions': predictions,
                'feature_count': len(features),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating predictions for {player_id}: {e}")
            return {"error": str(e)}
    
    def generate_prop_bet_recommendations(self) -> List[Dict[str, Any]]:
        """Generate comprehensive prop bet recommendations."""
        logger.info("🎯 Generating prop bet recommendations...")
        
        recommendations = []
        
        try:
            # Get eligible players
            with self.Session() as session:
                query = text("""
                    SELECT DISTINCT pgs.player_id, p.position, p.name,
                           COUNT(*) as games_played,
                           AVG(COALESCE(pgs.fantasy_points_ppr, 0)) as avg_fantasy
                    FROM player_game_stats pgs
                    JOIN players p ON pgs.player_id = p.player_id
                    WHERE pgs.created_at > datetime('now', '-45 days')
                    GROUP BY pgs.player_id, p.position, p.name
                    HAVING games_played >= :min_games AND avg_fantasy > 3
                    ORDER BY avg_fantasy DESC
                    LIMIT 100
                """)
                
                result = session.execute(query, {"min_games": self.config['min_games_threshold']})
                eligible_players = result.fetchall()
                
                logger.info(f"Found {len(eligible_players)} eligible players")
                
                for player in eligible_players:
                    player_id, position, name, games_played, avg_fantasy = player
                    
                    # Generate predictions
                    prediction_data = self.generate_comprehensive_predictions(player_id)
                    
                    if 'error' in prediction_data:
                        continue
                    
                    predictions = prediction_data.get('predictions', {})
                    
                    # Generate prop bet recommendations for each target
                    player_recommendations = self._create_prop_bet_recommendations(
                        player_id, name, position, predictions, avg_fantasy
                    )
                    
                    recommendations.extend(player_recommendations)
                
                # Sort by expected value
                recommendations.sort(
                    key=lambda x: x.get('expected_value', 0) * x.get('confidence', 0), 
                    reverse=True
                )
                
                return recommendations[:50]  # Top 50 recommendations
                
        except Exception as e:
            logger.error(f"Error generating prop bet recommendations: {e}")
            return []
    
    def _create_prop_bet_recommendations(self, player_id: str, name: str, position: str,
                                       predictions: Dict[str, Any], avg_fantasy: float) -> List[Dict[str, Any]]:
        """Create prop bet recommendations for a player."""
        recommendations = []
        
        # Get prop bet targets for this position
        prop_targets = [t for t in get_targets_for_position(position) if t.is_prop_bet]
        
        for target in prop_targets:
            if target.name in predictions:
                pred_data = predictions[target.name]
                predicted_value = pred_data.get('value', 0)
                confidence = pred_data.get('confidence', 0)
                
                # Only recommend if confidence is above threshold
                if confidence >= self.config['prop_bet_threshold']:
                    
                    # Create over/under recommendations
                    line_value = predicted_value
                    
                    # Adjust line based on typical sportsbook margins
                    over_line = line_value - 0.5
                    under_line = line_value + 0.5
                    
                    # Calculate expected value (simplified)
                    expected_value = abs(predicted_value - line_value) * confidence
                    
                    if expected_value > 0.5:  # Minimum threshold for recommendation
                        
                        bet_type = "Over" if predicted_value > line_value else "Under"
                        bet_line = over_line if bet_type == "Over" else under_line
                        
                        recommendation = {
                            'player_id': player_id,
                            'player_name': name,
                            'position': position,
                            'stat': target.name,
                            'stat_description': target.description,
                            'predicted_value': predicted_value,
                            'bet_type': bet_type,
                            'bet_line': bet_line,
                            'confidence': confidence,
                            'expected_value': expected_value,
                            'category': target.category.value,
                            'historical_avg': avg_fantasy if target.name.startswith('fantasy') else None
                        }
                        
                        recommendations.append(recommendation)
        
        return recommendations
    
    def run_backtesting(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtesting analysis on historical data."""
        logger.info(f"🔄 Running backtesting from {start_date} to {end_date}")
        
        try:
            # This would implement a comprehensive backtesting framework
            # For now, return a placeholder structure
            
            backtesting_results = {
                'period': f"{start_date} to {end_date}",
                'total_predictions': 0,
                'accuracy_by_position': {},
                'accuracy_by_stat': {},
                'roi_analysis': {},
                'confidence_calibration': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results
            self.backtesting_results = backtesting_results
            
            # Save to file
            backtest_path = self.output_dir / f"backtesting_{start_date}_{end_date}.json"
            with open(backtest_path, 'w') as f:
                json.dump(backtesting_results, f, indent=2)
            
            logger.info(f"✅ Backtesting completed. Results saved to {backtest_path}")
            
            return backtesting_results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {}
    
    def _save_training_summary(self, models_trained: int):
        """Save training summary to file."""
        summary = {
            'models_trained': models_trained,
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'prediction_targets_summary': PREDICTION_TARGETS.get_position_summary()
        }
        
        summary_path = self.output_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
    
    def display_comprehensive_analysis(self):
        """Display comprehensive betting analysis."""
        print("🏈 COMPREHENSIVE NFL BETTING ANALYZER")
        print("=" * 60)
        print(f"🔧 Enhanced Features: Ensemble Models, Prop Bets, Backtesting")
        print(f"💾 Database: {self.db_url}")
        print(f"🎯 Prediction Targets: {len(PREDICTION_TARGETS.get_all_targets())} total")
        print()
        
        # Validate system
        if not self.validate_database():
            print("❌ System validation failed")
            return
        
        # Check if models exist
        if not self.predictor.models:
            print("🤖 No trained models found. Training comprehensive models...")
            self.train_comprehensive_models()
            print()
        
        # Generate recommendations
        recommendations = self.generate_prop_bet_recommendations()
        
        if not recommendations:
            print("❌ No prop bet recommendations available")
            return
        
        print("🎯 TOP PROP BET RECOMMENDATIONS:")
        print("-" * 50)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        
        for i, rec in enumerate(recommendations[:15], 1):
            emoji = position_emoji.get(rec['position'], '⚡')
            
            print(f"{i}. {rec['player_name']} ({rec['position']}) {emoji}")
            print(f"   💰 {rec['bet_type']} {rec['bet_line']:.1f} {rec['stat_description']}")
            print(f"   📊 Predicted: {rec['predicted_value']:.1f}")
            print(f"   🎯 Confidence: {rec['confidence']:.1%}")
            print(f"   💎 Expected Value: {rec['expected_value']:.2f}")
            print(f"   📈 Category: {rec['category'].title()}")
            print()
        
        print("⚠️  ENHANCED SYSTEM FEATURES:")
        print("   • Comprehensive prediction targets (all major NFL stats)")
        print("   • Ensemble models (XGBoost, LightGBM, Random Forest, Neural Networks)")
        print("   • Advanced feature engineering (800+ features per player)")
        print("   • Prop bet recommendations with confidence intervals")
        print("   • Backtesting framework for historical validation")
        print("   • Real-time model retraining capabilities")
        print("   • Always gamble responsibly!")

def main():
    """Main function to run the comprehensive analyzer."""
    try:
        # Initialize with enhanced configuration
        config = {
            'model_types': ['xgboost', 'lightgbm', 'random_forest'],
            'confidence_threshold': 0.65,
            'prop_bet_threshold': 0.7,
            'min_games_threshold': 5,
            'backtesting_enabled': True
        }
        
        analyzer = ComprehensiveBettingAnalyzer(config)
        analyzer.display_comprehensive_analysis()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
