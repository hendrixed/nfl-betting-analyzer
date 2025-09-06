#!/usr/bin/env python3
"""
Optimized NFL Prediction Pipeline
High-performance version with caching, batch processing, and parallel execution
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import redis
from sqlalchemy import create_engine, select, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

# Import our modules
from database_models import (
    Player, Team, Game, PlayerGameStats, BettingLine, 
    PlayerPrediction, GamePrediction, FeatureStore, ModelPerformance
)
from config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class OptimizedPipelineConfig:
    """Configuration for optimized prediction pipeline."""
    database_url: str
    
    # Performance settings
    max_workers: int = 8
    batch_size: int = 500
    prefetch_size: int = 1000
    
    # Caching settings
    enable_redis_cache: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600  # 1 hour
    
    # Feature caching
    feature_cache_enabled: bool = True
    feature_cache_size: int = 10000
    
    # Model caching
    model_cache_enabled: bool = True
    model_cache_size: int = 100
    
    # Database optimization
    connection_pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    
    # Prediction settings
    prediction_horizon_days: int = 7
    confidence_threshold: float = 0.6
    
    # Processing optimization
    use_multiprocessing: bool = True
    chunk_size: int = 100


class CacheManager:
    """Manages caching for the prediction pipeline."""
    
    def __init__(self, config: OptimizedPipelineConfig):
        self.config = config
        self.redis_client = None
        self.local_cache = {}
        
        if config.enable_redis_cache:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis cache unavailable: {e}")
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value.encode('latin1'))
            
            # Fallback to local cache
            return self.local_cache.get(key)
            
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.config.cache_ttl
            
            # Try Redis first
            if self.redis_client:
                serialized = pickle.dumps(value).decode('latin1')
                return self.redis_client.setex(key, ttl, serialized)
            
            # Fallback to local cache
            self.local_cache[key] = value
            return True
            
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            
            self.local_cache.pop(key, None)
            return True
            
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache."""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
            
            self.local_cache.clear()
            return True
            
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
            return False


class OptimizedFeatureEngine:
    """Optimized feature engineering with caching."""
    
    def __init__(self, config: OptimizedPipelineConfig, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self.feature_cache = {}
        
    @lru_cache(maxsize=1000)
    def get_player_features_cached(self, player_id: str, game_id: str) -> Optional[Dict]:
        """Get cached player features."""
        cache_key = f"features:{player_id}:{game_id}"
        return self.cache.get(cache_key)
    
    def batch_engineer_features(self, player_game_pairs: List[Tuple[str, str]]) -> Dict[str, Dict]:
        """Engineer features for multiple player-game pairs in batch."""
        results = {}
        
        # Group by game for efficient processing
        games_dict = {}
        for player_id, game_id in player_game_pairs:
            if game_id not in games_dict:
                games_dict[game_id] = []
            games_dict[game_id].append(player_id)
        
        # Process each game's players together
        for game_id, player_ids in games_dict.items():
            game_features = self._engineer_game_features_batch(game_id, player_ids)
            results.update(game_features)
        
        return results
    
    def _engineer_game_features_batch(self, game_id: str, player_ids: List[str]) -> Dict[str, Dict]:
        """Engineer features for all players in a game efficiently."""
        try:
            # Load game context once
            game_context = self._get_game_context(game_id)
            
            results = {}
            for player_id in player_ids:
                cache_key = f"features:{player_id}:{game_id}"
                
                # Check cache first
                cached_features = self.cache.get(cache_key)
                if cached_features:
                    results[f"{player_id}:{game_id}"] = cached_features
                    continue
                
                # Engineer features
                features = self._engineer_player_features_optimized(
                    player_id, game_id, game_context
                )
                
                if features:
                    results[f"{player_id}:{game_id}"] = features
                    # Cache the result
                    self.cache.set(cache_key, features)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch feature engineering failed for game {game_id}: {e}")
            return {}
    
    def _get_game_context(self, game_id: str) -> Dict:
        """Get game context information (cached)."""
        cache_key = f"game_context:{game_id}"
        
        cached_context = self.cache.get(cache_key)
        if cached_context:
            return cached_context
        
        # Load game context from database
        # This would contain weather, opponent stats, etc.
        context = {
            'game_id': game_id,
            'weather': {'temperature': 70, 'wind': 5, 'precipitation': 0},
            'opponent_defense_ranks': {},
            'game_importance': 1.0
        }
        
        self.cache.set(cache_key, context)
        return context
    
    def _engineer_player_features_optimized(self, player_id: str, game_id: str, game_context: Dict) -> Dict:
        """Engineer features for a single player with optimization."""
        try:
            # This would contain the actual feature engineering logic
            # For now, return mock features
            features = {
                'recent_form_avg': np.random.normal(15, 5),
                'opponent_matchup_score': np.random.normal(0, 1),
                'home_away_factor': 1.0 if game_context.get('is_home') else 0.9,
                'weather_impact': self._calculate_weather_impact(game_context.get('weather', {})),
                'rest_days': 7,
                'injury_risk': 0.1,
                'target_share': 0.25,
                'red_zone_efficiency': 0.6
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature engineering failed for {player_id}: {e}")
            return {}
    
    def _calculate_weather_impact(self, weather: Dict) -> float:
        """Calculate weather impact on performance."""
        if not weather:
            return 1.0
        
        temp_factor = 1.0
        wind_factor = max(0.8, 1.0 - weather.get('wind', 0) * 0.02)
        precip_factor = max(0.9, 1.0 - weather.get('precipitation', 0) * 0.1)
        
        return temp_factor * wind_factor * precip_factor


class OptimizedPredictor:
    """Optimized prediction engine with model caching."""
    
    def __init__(self, config: OptimizedPipelineConfig, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self.model_cache = {}
        
    @lru_cache(maxsize=100)
    def get_model_cached(self, position: str, target: str) -> Optional[Any]:
        """Get cached model."""
        cache_key = f"model:{position}:{target}"
        return self.cache.get(cache_key)
    
    def batch_predict(self, feature_batches: List[Dict]) -> List[Dict]:
        """Make predictions for multiple feature sets in batch."""
        results = []
        
        # Group by position for efficient model loading
        position_groups = {}
        for i, features in enumerate(feature_batches):
            position = features.get('position', 'QB')
            if position not in position_groups:
                position_groups[position] = []
            position_groups[position].append((i, features))
        
        # Process each position group
        for position, feature_list in position_groups.items():
            position_results = self._predict_position_batch(position, feature_list)
            results.extend(position_results)
        
        # Sort results back to original order
        results.sort(key=lambda x: x['original_index'])
        return [r['prediction'] for r in results]
    
    def _predict_position_batch(self, position: str, feature_list: List[Tuple[int, Dict]]) -> List[Dict]:
        """Make predictions for a batch of features for one position."""
        results = []
        
        try:
            # Load models for this position (cached)
            models = self._load_position_models(position)
            
            for original_index, features in feature_list:
                prediction = self._make_single_prediction(features, models)
                results.append({
                    'original_index': original_index,
                    'prediction': prediction
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed for {position}: {e}")
            return []
    
    def _load_position_models(self, position: str) -> Dict:
        """Load models for a position (cached)."""
        cache_key = f"models:{position}"
        
        cached_models = self.cache.get(cache_key)
        if cached_models:
            return cached_models
        
        # Mock model loading - in reality would load from files
        models = {
            'fantasy_points': {'type': 'xgboost', 'accuracy': 0.85},
            'passing_yards': {'type': 'lightgbm', 'accuracy': 0.82},
            'rushing_yards': {'type': 'random_forest', 'accuracy': 0.78},
            'receiving_yards': {'type': 'neural_net', 'accuracy': 0.80}
        }
        
        self.cache.set(cache_key, models, ttl=7200)  # Cache for 2 hours
        return models
    
    def _make_single_prediction(self, features: Dict, models: Dict) -> Dict:
        """Make prediction for single feature set."""
        try:
            # Mock prediction logic
            predictions = {}
            
            for target, model in models.items():
                base_value = features.get('recent_form_avg', 15)
                noise = np.random.normal(0, 2)
                predicted_value = max(0, base_value + noise)
                
                predictions[target] = {
                    'predicted_value': predicted_value,
                    'confidence': model['accuracy'] + np.random.normal(0, 0.05),
                    'model_type': model['type']
                }
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Single prediction failed: {e}")
            return {}


class OptimizedNFLPipeline:
    """Optimized NFL prediction pipeline."""
    
    def __init__(self, config: OptimizedPipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache manager
        self.cache = CacheManager(config)
        
        # Initialize optimized components
        self.feature_engine = OptimizedFeatureEngine(config, self.cache)
        self.predictor = OptimizedPredictor(config, self.cache)
        
        # Database setup with connection pooling
        self.engine = create_engine(
            config.database_url,
            poolclass=QueuePool,
            pool_size=config.connection_pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_pre_ping=True
        )
        self.Session = sessionmaker(bind=self.engine)
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_prediction_time': 0,
            'batch_processing_time': 0
        }
    
    async def run_optimized_pipeline(self) -> Dict[str, Any]:
        """Run the optimized prediction pipeline."""
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting optimized prediction pipeline...")
            
            # Step 1: Get upcoming games and players efficiently
            upcoming_data = await self._get_upcoming_data_batch()
            
            # Step 2: Batch feature engineering
            features_start = time.time()
            feature_results = await self._batch_engineer_features(upcoming_data)
            features_time = time.time() - features_start
            
            # Step 3: Batch predictions
            predictions_start = time.time()
            prediction_results = await self._batch_make_predictions(feature_results)
            predictions_time = time.time() - predictions_start
            
            # Step 4: Batch save results
            save_start = time.time()
            await self._batch_save_predictions(prediction_results)
            save_time = time.time() - save_start
            
            total_time = time.time() - start_time
            
            # Update metrics
            self.metrics.update({
                'total_predictions': len(prediction_results),
                'feature_engineering_time': features_time,
                'prediction_time': predictions_time,
                'save_time': save_time,
                'total_pipeline_time': total_time,
                'predictions_per_second': len(prediction_results) / total_time if total_time > 0 else 0
            })
            
            self.logger.info(f"✅ Pipeline completed: {len(prediction_results)} predictions in {total_time:.2f}s")
            self.logger.info(f"📊 Performance: {self.metrics['predictions_per_second']:.1f} predictions/sec")
            
            return {
                'success': True,
                'predictions_made': len(prediction_results),
                'execution_time': total_time,
                'metrics': self.metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optimized pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': self.metrics
            }
    
    async def _get_upcoming_data_batch(self) -> Dict[str, Any]:
        """Get upcoming games and players in batch."""
        try:
            with self.Session() as session:
                # Get upcoming games
                end_date = date.today() + timedelta(days=self.config.prediction_horizon_days)
                
                games = session.query(Game).filter(
                    and_(
                        Game.game_date >= date.today(),
                        Game.game_date <= end_date,
                        Game.game_status == 'scheduled'
                    )
                ).limit(self.config.prefetch_size).all()
                
                # Get active players
                players = session.query(Player).filter(
                    Player.is_active == True
                ).limit(self.config.prefetch_size).all()
                
                # Create player-game pairs
                player_game_pairs = []
                for game in games:
                    game_players = [p for p in players if p.current_team in [game.home_team, game.away_team]]
                    for player in game_players:
                        if player.position in ['QB', 'RB', 'WR', 'TE']:
                            player_game_pairs.append((player.player_id, game.game_id))
                
                return {
                    'games': [{'game_id': g.game_id, 'home_team': g.home_team, 'away_team': g.away_team} for g in games],
                    'players': [{'player_id': p.player_id, 'position': p.position, 'team': p.current_team} for p in players],
                    'player_game_pairs': player_game_pairs
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get upcoming data: {e}")
            return {'games': [], 'players': [], 'player_game_pairs': []}
    
    async def _batch_engineer_features(self, upcoming_data: Dict) -> List[Dict]:
        """Engineer features in batch with parallel processing."""
        try:
            player_game_pairs = upcoming_data['player_game_pairs']
            
            if not player_game_pairs:
                return []
            
            # Split into chunks for parallel processing
            chunks = [
                player_game_pairs[i:i + self.config.chunk_size]
                for i in range(0, len(player_game_pairs), self.config.chunk_size)
            ]
            
            all_features = []
            
            if self.config.use_multiprocessing and len(chunks) > 1:
                # Use multiprocessing for large batches
                with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = [
                        executor.submit(self.feature_engine.batch_engineer_features, chunk)
                        for chunk in chunks
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            chunk_features = future.result()
                            all_features.extend(chunk_features.values())
                        except Exception as e:
                            self.logger.warning(f"Feature chunk processing failed: {e}")
            else:
                # Use threading for smaller batches
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = [
                        executor.submit(self.feature_engine.batch_engineer_features, chunk)
                        for chunk in chunks
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            chunk_features = future.result()
                            all_features.extend(chunk_features.values())
                        except Exception as e:
                            self.logger.warning(f"Feature chunk processing failed: {e}")
            
            self.logger.info(f"✅ Engineered features for {len(all_features)} player-game combinations")
            return all_features
            
        except Exception as e:
            self.logger.error(f"Batch feature engineering failed: {e}")
            return []
    
    async def _batch_make_predictions(self, feature_results: List[Dict]) -> List[Dict]:
        """Make predictions in batch."""
        try:
            if not feature_results:
                return []
            
            # Split into batches
            batches = [
                feature_results[i:i + self.config.batch_size]
                for i in range(0, len(feature_results), self.config.batch_size)
            ]
            
            all_predictions = []
            
            for batch in batches:
                try:
                    batch_predictions = self.predictor.batch_predict(batch)
                    all_predictions.extend(batch_predictions)
                except Exception as e:
                    self.logger.warning(f"Prediction batch failed: {e}")
                    continue
            
            self.logger.info(f"✅ Generated {len(all_predictions)} predictions")
            return all_predictions
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return []
    
    async def _batch_save_predictions(self, predictions: List[Dict]):
        """Save predictions to database in batch."""
        try:
            if not predictions:
                return
            
            # Prepare batch insert data
            prediction_records = []
            
            for i, prediction in enumerate(predictions):
                # Mock player and game IDs - in reality would come from features
                player_id = f"player_{i % 100}"
                game_id = f"game_{i % 10}"
                
                record = {
                    'player_id': player_id,
                    'game_id': game_id,
                    'model_version': 'v2.0_optimized',
                    'model_type': 'ensemble_optimized',
                    'predicted_fantasy_points': prediction.get('fantasy_points', {}).get('predicted_value', 0),
                    'confidence_overall': np.mean([p.get('confidence', 0.7) for p in prediction.values()]),
                    'prediction_timestamp': datetime.now()
                }
                prediction_records.append(record)
            
            # Batch insert to database
            with self.Session() as session:
                # Use bulk insert for better performance
                session.execute(
                    PlayerPrediction.__table__.insert(),
                    prediction_records
                )
                session.commit()
            
            self.logger.info(f"✅ Saved {len(prediction_records)} predictions to database")
            
        except Exception as e:
            self.logger.error(f"Batch save failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        cache_hit_rate = (
            self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
            if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        )
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_efficiency': 'High' if cache_hit_rate > 0.8 else 'Medium' if cache_hit_rate > 0.5 else 'Low'
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear()
        self.logger.info("🧹 All caches cleared")


# Factory function for easy initialization
def create_optimized_pipeline(database_url: str = None) -> OptimizedNFLPipeline:
    """Create an optimized pipeline with default configuration."""
    
    if not database_url:
        config = get_config()
        database_url = f"sqlite:///{config.database.path}"
    
    pipeline_config = OptimizedPipelineConfig(
        database_url=database_url,
        max_workers=8,
        batch_size=500,
        enable_redis_cache=True,
        feature_cache_enabled=True,
        model_cache_enabled=True,
        use_multiprocessing=True
    )
    
    return OptimizedNFLPipeline(pipeline_config)


async def main():
    """Example usage of optimized pipeline."""
    
    print("🚀 NFL OPTIMIZED PREDICTION PIPELINE")
    print("=" * 50)
    
    try:
        # Create optimized pipeline
        pipeline = create_optimized_pipeline()
        
        # Run the pipeline
        results = await pipeline.run_optimized_pipeline()
        
        if results['success']:
            print(f"✅ Pipeline completed successfully!")
            print(f"📊 Predictions made: {results['predictions_made']}")
            print(f"⏱️  Execution time: {results['execution_time']:.2f}s")
            
            # Show performance metrics
            metrics = pipeline.get_performance_metrics()
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   Predictions/sec: {metrics.get('predictions_per_second', 0):.1f}")
            print(f"   Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")
            print(f"   Cache efficiency: {metrics.get('cache_efficiency', 'Unknown')}")
        else:
            print(f"❌ Pipeline failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
